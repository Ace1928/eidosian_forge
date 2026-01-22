import functools
import heapq
import itertools
import random
from collections import Counter, OrderedDict, defaultdict
import numpy as np
from . import helpers
class DynamicProgramming(PathOptimizer):
    """
    Finds the optimal path of pairwise contractions without intermediate outer
    products based a dynamic programming approach presented in
    Phys. Rev. E 90, 033315 (2014) (the corresponding preprint is publically
    available at https://arxiv.org/abs/1304.6112). This method is especially
    well-suited in the area of tensor network states, where it usually
    outperforms all the other optimization strategies.

    This algorithm shows exponential scaling with the number of inputs
    in the worst case scenario (see example below). If the graph to be
    contracted consists of disconnected subgraphs, the algorithm scales
    linearly in the number of disconnected subgraphs and only exponentially
    with the number of inputs per subgraph.

    Parameters
    ----------
    minimize : {'flops', 'size'}, optional
        Whether to find the contraction that minimizes the number of
        operations or the size of the largest intermediate tensor.
    cost_cap : {True, False, int}, optional
        How to implement cost-capping:

            * True - iteratively increase the cost-cap
            * False - implement no cost-cap at all
            * int - use explicit cost cap

    search_outer : bool, optional
        In rare circumstances the optimal contraction may involve an outer
        product, this option allows searching such contractions but may well
        slow down the path finding considerably on all but very small graphs.
    """

    def __init__(self, minimize='flops', cost_cap=True, search_outer=False):
        self.minimize = minimize
        self._check_contraction = {'flops': _dp_compare_flops, 'size': _dp_compare_size}[self.minimize]
        self.search_outer = search_outer
        self._check_outer = {False: lambda x: x, True: lambda x: True}[self.search_outer]
        self.cost_cap = cost_cap

    def __call__(self, inputs, output, size_dict, memory_limit=None):
        """
        Parameters
        ----------
        inputs : list
            List of sets that represent the lhs side of the einsum subscript
        output : set
            Set that represents the rhs side of the overall einsum subscript
        size_dict : dictionary
            Dictionary of index sizes
        memory_limit : int
            The maximum number of elements in a temporary array

        Returns
        -------
        path : list
            The contraction order (a list of tuples of ints).

        Examples
        --------
        >>> n_in = 3  # exponential scaling
        >>> n_out = 2 # linear scaling
        >>> s = dict()
        >>> i_all = []
        >>> for _ in range(n_out):
        >>>     i = [set() for _ in range(n_in)]
        >>>     for j in range(n_in):
        >>>         for k in range(j+1, n_in):
        >>>             c = oe.get_symbol(len(s))
        >>>             i[j].add(c)
        >>>             i[k].add(c)
        >>>             s[c] = 2
        >>>     i_all.extend(i)
        >>> o = DynamicProgramming()
        >>> o(i_all, set(), s)
        [(1, 2), (0, 4), (1, 2), (0, 2), (0, 1)]
        """
        ind_counts = Counter(itertools.chain(*inputs, output))
        all_inds = tuple(ind_counts)
        symbol2int = {c: j for j, c in enumerate(all_inds)}
        inputs = [set((symbol2int[c] for c in i)) for i in inputs]
        output = set((symbol2int[c] for c in output))
        size_dict = {symbol2int[c]: v for c, v in size_dict.items() if c in symbol2int}
        size_dict = [size_dict[j] for j in range(len(size_dict))]
        inputs, inputs_done, inputs_contractions = _dp_parse_out_single_term_ops(inputs, all_inds, ind_counts)
        if not inputs:
            return _tree_to_sequence(simple_tree_tuple(inputs_done))
        subgraph_contractions = inputs_done
        subgraph_contractions_size = [1] * len(inputs_done)
        if self.search_outer:
            subgraphs = [set(range(len(inputs)))]
        else:
            subgraphs = _find_disconnected_subgraphs(inputs, output)
        all_tensors = (1 << len(inputs)) - 1
        for g in subgraphs:
            x = [None] * 2 + [dict() for j in range(len(g) - 1)]
            x[1] = OrderedDict(((1 << j, (inputs[j], 0, inputs_contractions[j])) for j in g))
            g = functools.reduce(lambda x, y: x | y, (1 << j for j in g))
            subgraph_inds = set.union(*_bitmap_select(g, inputs))
            if self.cost_cap is True:
                cost_cap = helpers.compute_size_by_dict(subgraph_inds & output, size_dict)
            elif self.cost_cap is False:
                cost_cap = float('inf')
            else:
                cost_cap = self.cost_cap
            cost_increment = max(min(map(size_dict.__getitem__, subgraph_inds)), 2)
            while len(x[-1]) == 0:
                for n in range(2, len(x[1]) + 1):
                    xn = x[n]
                    for m in range(1, n // 2 + 1):
                        for s1, (i1, cost1, cntrct1) in x[m].items():
                            for s2, (i2, cost2, cntrct2) in x[n - m].items():
                                if not s1 & s2 and (m != n - m or s1 < s2):
                                    i1_cut_i2_wo_output = (i1 & i2) - output
                                    if self._check_outer(i1_cut_i2_wo_output):
                                        i1_union_i2 = i1 | i2
                                        self._check_contraction(cost1, cost2, i1_union_i2, size_dict, cost_cap, s1, s2, xn, g, all_tensors, inputs, i1_cut_i2_wo_output, memory_limit, cntrct1, cntrct2)
                cost_cap = cost_increment * cost_cap
            i, cost, contraction = list(x[-1].values())[0]
            subgraph_contractions.append(contraction)
            subgraph_contractions_size.append(helpers.compute_size_by_dict(i, size_dict))
        subgraph_contractions = [subgraph_contractions[j] for j in sorted(range(len(subgraph_contractions_size)), key=subgraph_contractions_size.__getitem__)]
        tree = simple_tree_tuple(subgraph_contractions)
        return _tree_to_sequence(tree)