import functools
import heapq
import math
import numbers
import time
from collections import deque
from . import helpers, paths
class RandomGreedy(RandomOptimizer):
    """

    Parameters
    ----------
    cost_fn : callable, optional
        A function that returns a heuristic 'cost' of a potential contraction
        with which to sort candidates. Should have signature
        ``cost_fn(size12, size1, size2, k12, k1, k2)``.
    temperature : float, optional
        When choosing a possible contraction, its relative probability will be
        proportional to ``exp(-cost / temperature)``. Thus the larger
        ``temperature`` is, the further random paths will stray from the normal
        'greedy' path. Conversely, if set to zero, only paths with exactly the
        same cost as the best at each step will be explored.
    rel_temperature : bool, optional
        Whether to normalize the ``temperature`` at each step to the scale of
        the best cost. This is generally beneficial as the magnitude of costs
        can vary significantly throughout a contraction. If False, the
        algorithm will end up branching when the absolute cost is low, but
        stick to the 'greedy' path when the cost is high - this can also be
        beneficial.
    nbranch : int, optional
        How many potential paths to calculate probability for and choose from
        at each step.
    kwargs
        Supplied to RandomOptimizer.

    See Also
    --------
    RandomOptimizer
    """

    def __init__(self, cost_fn='memory-removed-jitter', temperature=1.0, rel_temperature=True, nbranch=8, **kwargs):
        self.cost_fn = cost_fn
        self.temperature = temperature
        self.rel_temperature = rel_temperature
        self.nbranch = nbranch
        super().__init__(**kwargs)

    @property
    def choose_fn(self):
        """The function that chooses which contraction to take - make this a
        property so that ``temperature`` and ``nbranch`` etc. can be updated
        between runs.
        """
        if self.nbranch == 1:
            return None
        return functools.partial(thermal_chooser, temperature=self.temperature, nbranch=self.nbranch, rel_temperature=self.rel_temperature)

    def setup(self, inputs, output, size_dict):
        fn = _trial_greedy_ssa_path_and_cost
        args = (inputs, output, size_dict, self.choose_fn, self.cost_fn)
        return (fn, args)