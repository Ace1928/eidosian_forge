import collections
import random
import threading
import time
from contextlib import contextmanager
from typing import Any, Callable, Generic, Iterable, List, TypeVar
import ray
from ray.util.annotations import Deprecated
from ray.util.iter_metrics import MetricsContext, SharedMetrics
@Deprecated
class LocalIterator(Generic[T]):
    """An iterator over a single shard of data.

    It implements similar transformations as ParallelIterator[T], but the
    transforms will be applied locally and not remotely in parallel.

    This class is **serializable** and can be passed to other remote
    tasks and actors. However, it should be read from at most one process at
    a time."""
    ON_FETCH_START_HOOK_NAME = '_on_fetch_start'
    thread_local = threading.local()

    def __init__(self, base_iterator: Callable[[], Iterable[T]], shared_metrics: SharedMetrics, local_transforms: List[Callable[[Iterable], Any]]=None, timeout: int=None, name=None):
        """Create a local iterator (this is an internal function).

        Args:
            base_iterator: A function that produces the base iterator.
                This is a function so that we can ensure LocalIterator is
                serializable.
            shared_metrics: Existing metrics context or a new
                context. Should be the same for each chained iterator.
            local_transforms: A list of transformation functions to be
                applied on top of the base iterator. When iteration begins, we
                create the base iterator and apply these functions. This lazy
                creation ensures LocalIterator is serializable until you start
                iterating over it.
            timeout: Optional timeout in seconds for this iterator, after
                which _NextValueNotReady will be returned. This avoids
                blocking.
            name: Optional name for this iterator.
        """
        assert isinstance(shared_metrics, SharedMetrics)
        self.base_iterator = base_iterator
        self.built_iterator = None
        self.local_transforms = local_transforms or []
        self.shared_metrics = shared_metrics
        self.timeout = timeout
        self.name = name or 'unknown'

    @staticmethod
    def get_metrics() -> MetricsContext:
        """Return the current metrics context.

        This can only be called within an iterator function."""
        if not hasattr(LocalIterator.thread_local, 'metrics') or LocalIterator.thread_local.metrics is None:
            raise ValueError('Cannot access context outside an iterator.')
        return LocalIterator.thread_local.metrics

    def _build_once(self):
        if self.built_iterator is None:
            it = iter(self.base_iterator(self.timeout))
            for fn in self.local_transforms:
                it = fn(it)
            self.built_iterator = it

    @contextmanager
    def _metrics_context(self):
        self.thread_local.metrics = self.shared_metrics.get()
        yield

    def __iter__(self):
        self._build_once()
        return self.built_iterator

    def __next__(self):
        self._build_once()
        return next(self.built_iterator)

    def __str__(self):
        return repr(self)

    def __repr__(self):
        return f'LocalIterator[{self.name}]'

    def transform(self, fn: Callable[[Iterable[T]], Iterable[U]]) -> 'LocalIterator[U]':

        def apply_transform(it):
            for item in fn(it):
                yield item
        return LocalIterator(self.base_iterator, self.shared_metrics, self.local_transforms + [apply_transform], name=self.name + '.transform()')

    def for_each(self, fn: Callable[[T], U], max_concurrency=1, resources=None) -> 'LocalIterator[U]':
        if max_concurrency == 1:

            def apply_foreach(it):
                for item in it:
                    if isinstance(item, _NextValueNotReady):
                        yield item
                    else:
                        while True:
                            with self._metrics_context():
                                result = fn(item)
                            yield result
                            if not isinstance(result, _NextValueNotReady):
                                break
        else:
            if resources is None:
                resources = {}

            def apply_foreach(it):
                cur = []
                remote = ray.remote(fn).options(**resources)
                remote_fn = remote.remote
                for item in it:
                    if isinstance(item, _NextValueNotReady):
                        yield item
                    else:
                        if max_concurrency and len(cur) >= max_concurrency:
                            finished, cur = ray.wait(cur)
                            yield from ray.get(finished)
                        cur.append(remote_fn(item))
                while cur:
                    finished, cur = ray.wait(cur)
                    yield from ray.get(finished)
        if hasattr(fn, LocalIterator.ON_FETCH_START_HOOK_NAME):
            unwrapped = apply_foreach

            def add_wait_hooks(it):
                it = unwrapped(it)
                new_item = True
                while True:
                    if new_item:
                        with self._metrics_context():
                            fn._on_fetch_start()
                        new_item = False
                    item = next(it)
                    if not isinstance(item, _NextValueNotReady):
                        new_item = True
                    yield item
            apply_foreach = add_wait_hooks
        return LocalIterator(self.base_iterator, self.shared_metrics, self.local_transforms + [apply_foreach], name=self.name + '.for_each()')

    def filter(self, fn: Callable[[T], bool]) -> 'LocalIterator[T]':

        def apply_filter(it):
            for item in it:
                with self._metrics_context():
                    if isinstance(item, _NextValueNotReady) or fn(item):
                        yield item
        return LocalIterator(self.base_iterator, self.shared_metrics, self.local_transforms + [apply_filter], name=self.name + '.filter()')

    def batch(self, n: int) -> 'LocalIterator[List[T]]':

        def apply_batch(it):
            batch = []
            for item in it:
                if isinstance(item, _NextValueNotReady):
                    yield item
                else:
                    batch.append(item)
                    if len(batch) >= n:
                        yield batch
                        batch = []
            if batch:
                yield batch
        return LocalIterator(self.base_iterator, self.shared_metrics, self.local_transforms + [apply_batch], name=self.name + f'.batch({n})')

    def flatten(self) -> 'LocalIterator[T[0]]':

        def apply_flatten(it):
            for item in it:
                if isinstance(item, _NextValueNotReady):
                    yield item
                else:
                    for subitem in item:
                        yield subitem
        return LocalIterator(self.base_iterator, self.shared_metrics, self.local_transforms + [apply_flatten], name=self.name + '.flatten()')

    def shuffle(self, shuffle_buffer_size: int, seed: int=None) -> 'LocalIterator[T]':
        """Shuffle items of this iterator

        Args:
            shuffle_buffer_size: The algorithm fills a buffer with
                shuffle_buffer_size elements and randomly samples elements from
                this buffer, replacing the selected elements with new elements.
                For perfect shuffling, this argument should be greater than or
                equal to the largest iterator size.
            seed: Seed to use for
                randomness. Default value is None.

        Returns:
            A new LocalIterator with shuffling applied
        """
        shuffle_random = random.Random(seed)

        def apply_shuffle(it):
            buffer = []
            for item in it:
                if isinstance(item, _NextValueNotReady):
                    yield item
                else:
                    buffer.append(item)
                    if len(buffer) >= shuffle_buffer_size:
                        yield buffer.pop(shuffle_random.randint(0, len(buffer) - 1))
            while len(buffer) > 0:
                yield buffer.pop(shuffle_random.randint(0, len(buffer) - 1))
        return LocalIterator(self.base_iterator, self.shared_metrics, self.local_transforms + [apply_shuffle], name=self.name + '.shuffle(shuffle_buffer_size={}, seed={})'.format(shuffle_buffer_size, str(seed) if seed is not None else 'None'))

    def combine(self, fn: Callable[[T], List[U]]) -> 'LocalIterator[U]':
        it = self.for_each(fn).flatten()
        it.name = self.name + '.combine()'
        return it

    def zip_with_source_actor(self):

        def zip_with_source(item):
            metrics = LocalIterator.get_metrics()
            if metrics.current_actor is None:
                raise ValueError('Could not identify source actor of item')
            return (metrics.current_actor, item)
        it = self.for_each(zip_with_source)
        it.name = self.name + '.zip_with_source_actor()'
        return it

    def take(self, n: int) -> List[T]:
        """Return up to the first n items from this iterator."""
        out = []
        for item in self:
            out.append(item)
            if len(out) >= n:
                break
        return out

    def show(self, n: int=20):
        """Print up to the first n items from this iterator."""
        i = 0
        for item in self:
            print(item)
            i += 1
            if i >= n:
                break

    def duplicate(self, n) -> List['LocalIterator[T]']:
        """Copy this iterator `n` times, duplicating the data.

        The child iterators will be prioritized by how much of the parent
        stream they have consumed. That is, we will not allow children to fall
        behind, since that can cause infinite memory buildup in this operator.

        Returns:
            List[LocalIterator[T]]: child iterators that each have a copy
                of the data of this iterator.
        """
        if n < 2:
            raise ValueError('Number of copies must be >= 2')
        queues = []
        for _ in range(n):
            queues.append(collections.deque())

        def fill_next(timeout):
            self.timeout = timeout
            item = next(self)
            for q in queues:
                q.append(item)

        def make_next(i):

            def gen(timeout):
                while True:
                    my_len = len(queues[i])
                    max_len = max((len(q) for q in queues))
                    if my_len < max_len:
                        yield _NextValueNotReady()
                    else:
                        if len(queues[i]) == 0:
                            try:
                                fill_next(timeout)
                            except StopIteration:
                                return
                        yield queues[i].popleft()
            return gen
        iterators = []
        for i in range(n):
            iterators.append(LocalIterator(make_next(i), self.shared_metrics, [], name=self.name + f'.duplicate[{i}]'))
        return iterators

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