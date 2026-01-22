import functools
import heapq
import math
import numbers
import time
from collections import deque
from . import helpers, paths
class RandomOptimizer(paths.PathOptimizer):
    """Base class for running any random path finder that benefits
    from repeated calling, possibly in a parallel fashion. Custom random
    optimizers should subclass this, and the ``setup`` method should be
    implemented with the following signature::

        def setup(self, inputs, output, size_dict):
            # custom preparation here ...
            return trial_fn, trial_args

    Where ``trial_fn`` itself should have the signature::

        def trial_fn(r, *trial_args):
            # custom computation of path here
            return ssa_path, cost, size

    Where ``r`` is the run number and could for example be used to seed a
    random number generator. See ``RandomGreedy`` for an example.


    Parameters
    ----------
    max_repeats : int, optional
        The maximum number of repeat trials to have.
    max_time : float, optional
        The maximum amount of time to run the algorithm for.
    minimize : {'flops', 'size'}, optional
        Whether to favour paths that minimize the total estimated flop-count or
        the size of the largest intermediate created.
    parallel : {bool, int, or executor-pool like}, optional
        Whether to parallelize the random trials, by default ``False``. If
        ``True``, use a ``concurrent.futures.ProcessPoolExecutor`` with the same
        number of processes as cores. If an integer is specified, use that many
        processes instead. Finally, you can supply a custom executor-pool which
        should have an API matching that of the python 3 standard library
        module ``concurrent.futures``. Namely, a ``submit`` method that returns
        ``Future`` objects, themselves with ``result`` and ``cancel`` methods.
    pre_dispatch : int, optional
        If running in parallel, how many jobs to pre-dispatch so as to avoid
        submitting all jobs at once. Should also be more than twice the number
        of workers to avoid under-subscription. Default: 128.

    Attributes
    ----------
    path : list[tuple[int]]
        The best path found so far.
    costs : list[int]
        The list of each trial's costs found so far.
    sizes : list[int]
        The list of each trial's largest intermediate size so far.

    See Also
    --------
    RandomGreedy
    """

    def __init__(self, max_repeats=32, max_time=None, minimize='flops', parallel=False, pre_dispatch=128):
        if minimize not in ('flops', 'size'):
            raise ValueError("`minimize` should be one of {'flops', 'size'}.")
        self.max_repeats = max_repeats
        self.max_time = max_time
        self.minimize = minimize
        self.better = paths.get_better_fn(minimize)
        self.parallel = parallel
        self.pre_dispatch = pre_dispatch
        self.costs = []
        self.sizes = []
        self.best = {'flops': float('inf'), 'size': float('inf')}
        self._repeats_start = 0

    @property
    def path(self):
        """The best path found so far.
        """
        return paths.ssa_to_linear(self.best['ssa_path'])

    @property
    def parallel(self):
        return self._parallel

    @parallel.setter
    def parallel(self, parallel):
        if getattr(self, '_managing_executor', False):
            self._executor.shutdown()
        self._parallel = parallel
        self._managing_executor = False
        if parallel is False:
            self._executor = None
            return
        if parallel is True:
            from concurrent.futures import ProcessPoolExecutor
            self._executor = ProcessPoolExecutor()
            self._managing_executor = True
            return
        if isinstance(parallel, numbers.Number):
            from concurrent.futures import ProcessPoolExecutor
            self._executor = ProcessPoolExecutor(parallel)
            self._managing_executor = True
            return
        self._executor = parallel

    def _gen_results_parallel(self, repeats, trial_fn, args):
        """Lazily generate results from an executor without submitting all jobs at once.
        """
        self._futures = deque()
        for r in repeats:
            if len(self._futures) < self.pre_dispatch:
                self._futures.append(self._executor.submit(trial_fn, r, *args))
                continue
            yield self._futures.popleft().result()
        while self._futures:
            yield self._futures.popleft().result()

    def _cancel_futures(self):
        if self._executor is not None:
            for f in self._futures:
                f.cancel()

    def setup(self, inputs, output, size_dict):
        raise NotImplementedError

    def __call__(self, inputs, output, size_dict, memory_limit):
        self._check_args_against_first_call(inputs, output, size_dict)
        if self.max_time is not None:
            t0 = time.time()
        trial_fn, trial_args = self.setup(inputs, output, size_dict)
        r_start = self._repeats_start + len(self.costs)
        r_stop = r_start + self.max_repeats
        repeats = range(r_start, r_stop)
        if self._executor is not None:
            trials = self._gen_results_parallel(repeats, trial_fn, trial_args)
        else:
            trials = (trial_fn(r, *trial_args) for r in repeats)
        for ssa_path, cost, size in trials:
            self.costs.append(cost)
            self.sizes.append(size)
            found_new_best = self.better(cost, size, self.best['flops'], self.best['size'])
            if found_new_best:
                self.best['flops'] = cost
                self.best['size'] = size
                self.best['ssa_path'] = ssa_path
            if self.max_time is not None and time.time() > t0 + self.max_time:
                break
        self._cancel_futures()
        return self.path

    def __del__(self):
        if getattr(self, '_managing_executor', False):
            self._executor.shutdown()