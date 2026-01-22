import functools
import heapq
import itertools
import random
from collections import Counter, OrderedDict, defaultdict
import numpy as np
from . import helpers
class BranchBound(PathOptimizer):
    """
    Explores possible pair contractions in a depth-first recursive manner like
    the ``optimal`` approach, but with extra heuristic early pruning of branches
    as well sieving by ``memory_limit`` and the best path found so far. Returns
    the lowest cost path. This algorithm still scales factorially with respect
    to the elements in the list ``input_sets`` if ``nbranch`` is not set, but it
    scales exponentially like ``nbranch**len(input_sets)`` otherwise.

    Parameters
    ----------
    nbranch : None or int, optional
        How many branches to explore at each contraction step. If None, explore
        all possible branches. If an integer, branch into this many paths at
        each step. Defaults to None.
    cutoff_flops_factor : float, optional
        If at any point, a path is doing this much worse than the best path
        found so far was, terminate it. The larger this is made, the more paths
        will be fully explored and the slower the algorithm. Defaults to 4.
    minimize : {'flops', 'size'}, optional
        Whether to optimize the path with regard primarily to the total
        estimated flop-count, or the size of the largest intermediate. The
        option not chosen will still be used as a secondary criterion.
    cost_fn : callable, optional
        A function that returns a heuristic 'cost' of a potential contraction
        with which to sort candidates. Should have signature
        ``cost_fn(size12, size1, size2, k12, k1, k2)``.
    """

    def __init__(self, nbranch=None, cutoff_flops_factor=4, minimize='flops', cost_fn='memory-removed'):
        self.nbranch = nbranch
        self.cutoff_flops_factor = cutoff_flops_factor
        self.minimize = minimize
        self.cost_fn = _COST_FNS.get(cost_fn, cost_fn)
        self.better = get_better_fn(minimize)
        self.best = {'flops': float('inf'), 'size': float('inf')}
        self.best_progress = defaultdict(lambda: float('inf'))

    @property
    def path(self):
        return ssa_to_linear(self.best['ssa_path'])

    def __call__(self, inputs, output, size_dict, memory_limit=None):
        """

        Parameters
        ----------
        input_sets : list
            List of sets that represent the lhs side of the einsum subscript
        output_set : set
            Set that represents the rhs side of the overall einsum subscript
        idx_dict : dictionary
            Dictionary of index sizes
        memory_limit : int
            The maximum number of elements in a temporary array

        Returns
        -------
        path : list
            The contraction order within the memory limit constraint.

        Examples
        --------
        >>> isets = [set('abd'), set('ac'), set('bdc')]
        >>> oset = set('')
        >>> idx_sizes = {'a': 1, 'b':2, 'c':3, 'd':4}
        >>> optimal(isets, oset, idx_sizes, 5000)
        [(0, 2), (0, 1)]
        """
        self._check_args_against_first_call(inputs, output, size_dict)
        inputs = tuple(map(frozenset, inputs))
        output = frozenset(output)
        size_cache = {k: helpers.compute_size_by_dict(k, size_dict) for k in inputs}
        result_cache = {}

        def _branch_iterate(path, inputs, remaining, flops, size):
            if len(remaining) == 1:
                self.best['size'] = size
                self.best['flops'] = flops
                self.best['ssa_path'] = path
                return

            def _assess_candidate(k1, k2, i, j):
                try:
                    k12, flops12 = result_cache[k1, k2]
                except KeyError:
                    k12, flops12 = result_cache[k1, k2] = calc_k12_flops(inputs, output, remaining, i, j, size_dict)
                try:
                    size12 = size_cache[k12]
                except KeyError:
                    size12 = size_cache[k12] = helpers.compute_size_by_dict(k12, size_dict)
                new_flops = flops + flops12
                new_size = max(size, size12)
                if not self.better(new_flops, new_size, self.best['flops'], self.best['size']):
                    return None
                if new_flops < self.best_progress[len(inputs)]:
                    self.best_progress[len(inputs)] = new_flops
                elif new_flops > self.cutoff_flops_factor * self.best_progress[len(inputs)]:
                    return None
                if memory_limit not in _UNLIMITED_MEM and size12 > memory_limit:
                    new_flops = flops + _compute_oversize_flops(inputs, remaining, output, size_dict)
                    if new_flops < self.best['flops']:
                        self.best['flops'] = new_flops
                        self.best['ssa_path'] = path + (tuple(remaining),)
                    return None
                size1, size2 = (size_cache[inputs[i]], size_cache[inputs[j]])
                cost = self.cost_fn(size12, size1, size2, k12, k1, k2)
                return (cost, flops12, new_flops, new_size, (i, j), k12)
            candidates = []
            for i, j in itertools.combinations(remaining, 2):
                if i > j:
                    i, j = (j, i)
                k1, k2 = (inputs[i], inputs[j])
                if k1.isdisjoint(k2):
                    continue
                candidate = _assess_candidate(k1, k2, i, j)
                if candidate:
                    heapq.heappush(candidates, candidate)
            if not candidates:
                for i, j in itertools.combinations(remaining, 2):
                    if i > j:
                        i, j = (j, i)
                    k1, k2 = (inputs[i], inputs[j])
                    candidate = _assess_candidate(k1, k2, i, j)
                    if candidate:
                        heapq.heappush(candidates, candidate)
            bi = 0
            while (self.nbranch is None or bi < self.nbranch) and candidates:
                _, _, new_flops, new_size, (i, j), k12 = heapq.heappop(candidates)
                _branch_iterate(path=path + ((i, j),), inputs=inputs + (k12,), remaining=remaining - {i, j} | {len(inputs)}, flops=new_flops, size=new_size)
                bi += 1
        _branch_iterate(path=(), inputs=inputs, remaining=set(range(len(inputs))), flops=0, size=0)
        return self.path