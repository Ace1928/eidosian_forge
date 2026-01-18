import collections
import random
import threading
import time
from contextlib import contextmanager
from typing import Any, Callable, Generic, Iterable, List, TypeVar
import ray
from ray.util.annotations import Deprecated
from ray.util.iter_metrics import MetricsContext, SharedMetrics
def union(self, *others: 'LocalIterator[T]', deterministic: bool=False, round_robin_weights: List[float]=None) -> 'LocalIterator[T]':
    """Return an iterator that is the union of this and the others.

        Args:
            deterministic: If deterministic=True, we alternate between
                reading from one iterator and the others. Otherwise we return
                items from iterators as they become ready.
            round_robin_weights: List of weights to use for round robin
                mode. For example, [2, 1] will cause the iterator to pull twice
                as many items from the first iterator as the second.
                [2, 1, "*"] will cause as many items to be pulled as possible
                from the third iterator without blocking. This overrides the
                deterministic flag.
        """
    for it in others:
        if not isinstance(it, LocalIterator):
            raise ValueError(f'other must be of type LocalIterator, got {type(it)}')
    active = []
    parent_iters = [self] + list(others)
    shared_metrics = SharedMetrics(parents=[p.shared_metrics for p in parent_iters])
    timeout = None if deterministic else 0
    if round_robin_weights:
        if len(round_robin_weights) != len(parent_iters):
            raise ValueError('Length of round robin weights must equal number of iterators total.')
        timeouts = [0 if w == '*' else None for w in round_robin_weights]
    else:
        timeouts = [timeout] * len(parent_iters)
        round_robin_weights = [1] * len(parent_iters)
    for i, it in enumerate(parent_iters):
        active.append(LocalIterator(it.base_iterator, shared_metrics, it.local_transforms, timeout=timeouts[i]))
    active = list(zip(round_robin_weights, active))

    def build_union(timeout=None):
        while True:
            for weight, it in list(active):
                if weight == '*':
                    max_pull = 100
                else:
                    max_pull = _randomized_int_cast(weight)
                try:
                    for _ in range(max_pull):
                        item = next(it)
                        if isinstance(item, _NextValueNotReady):
                            if timeout is not None:
                                yield item
                            break
                        else:
                            yield item
                except StopIteration:
                    active.remove((weight, it))
            if not active:
                break
    return LocalIterator(build_union, shared_metrics, [], name=f'LocalUnion[{self}, {', '.join(map(str, others))}]')