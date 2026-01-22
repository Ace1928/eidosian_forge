import collections
import logging
import math
from typing import Any, Callable, Dict, Iterable, List, Optional, Tuple, TypeVar, Union
import ray
from ray.data._internal.block_list import BlockList
from ray.data._internal.delegating_block_builder import DelegatingBlockBuilder
from ray.data._internal.execution.interfaces import TaskContext
from ray.data._internal.progress_bar import ProgressBar
from ray.data._internal.remote_fn import cached_remote_fn
from ray.data.block import (
from ray.data.context import DataContext
from ray.types import ObjectRef
from ray.util.annotations import DeveloperAPI, PublicAPI
@PublicAPI
class ActorPoolStrategy(ComputeStrategy):
    """Specify the compute strategy for a Dataset transform.

    ActorPoolStrategy specifies that an autoscaling pool of actors should be used
    for a given Dataset transform. This is useful for stateful setup of callable
    classes.

    For a fixed-sized pool of size ``n``, specify ``compute=ActorPoolStrategy(size=n)``.
    To autoscale from ``m`` to ``n`` actors, specify
    ``ActorPoolStrategy(min_size=m, max_size=n)``.

    To increase opportunities for pipelining task dependency prefetching with
    computation and avoiding actor startup delays, set max_tasks_in_flight_per_actor
    to 2 or greater; to try to decrease the delay due to queueing of tasks on the worker
    actors, set max_tasks_in_flight_per_actor to 1.
    """

    def __init__(self, legacy_min_size: Optional[int]=None, legacy_max_size: Optional[int]=None, *, size: Optional[int]=None, min_size: Optional[int]=None, max_size: Optional[int]=None, max_tasks_in_flight_per_actor: Optional[int]=None):
        """Construct ActorPoolStrategy for a Dataset transform.

        Args:
            size: Specify a fixed size actor pool of this size. It is an error to
                specify both `size` and `min_size` or `max_size`.
            min_size: The minimize size of the actor pool.
            max_size: The maximum size of the actor pool.
            max_tasks_in_flight_per_actor: The maximum number of tasks to concurrently
                send to a single actor worker. Increasing this will increase
                opportunities for pipelining task dependency prefetching with
                computation and avoiding actor startup delays, but will also increase
                queueing delay.
        """
        if legacy_min_size is not None or legacy_max_size is not None:
            raise ValueError('In Ray 2.5, ActorPoolStrategy requires min_size and max_size to be explicit kwargs.')
        if size:
            if size < 1:
                raise ValueError('size must be >= 1', size)
            if max_size is not None or min_size is not None:
                raise ValueError('min_size and max_size cannot be set at the same time as `size`')
            min_size = size
            max_size = size
        if min_size is not None and min_size < 1:
            raise ValueError('min_size must be >= 1', min_size)
        if max_size is not None:
            if min_size is None:
                min_size = 1
            if min_size > max_size:
                raise ValueError('min_size must be <= max_size', min_size, max_size)
        if max_tasks_in_flight_per_actor is not None and max_tasks_in_flight_per_actor < 1:
            raise ValueError('max_tasks_in_flight_per_actor must be >= 1, got: ', max_tasks_in_flight_per_actor)
        self.min_size = min_size or 1
        self.max_size = max_size or float('inf')
        self.max_tasks_in_flight_per_actor = max_tasks_in_flight_per_actor
        self.num_workers = 0
        self.ready_to_total_workers_ratio = 0.8

    def _apply(self, block_fn: BlockTransform, remote_args: dict, block_list: BlockList, clear_input_blocks: bool, name: Optional[str]=None, min_rows_per_block: Optional[int]=None, fn: Optional[UserDefinedFunction]=None, fn_args: Optional[Iterable[Any]]=None, fn_kwargs: Optional[Dict[str, Any]]=None, fn_constructor_args: Optional[Iterable[Any]]=None, fn_constructor_kwargs: Optional[Dict[str, Any]]=None) -> BlockList:
        """Note: this is not part of the Dataset public API."""
        assert not DataContext.get_current().new_execution_backend, 'Legacy backend off'
        if fn_args is None:
            fn_args = tuple()
        if fn_kwargs is None:
            fn_kwargs = {}
        if fn_constructor_args is None:
            fn_constructor_args = tuple()
        if fn_constructor_kwargs is None:
            fn_constructor_kwargs = {}
        if name is None:
            name = 'map'
        blocks_in: List[Tuple[ObjectRef[Block], BlockMetadata]] = block_list.get_blocks_with_metadata()
        if min_rows_per_block is None:
            min_rows_per_block = 0
        if not math.isinf(self.max_size):
            total_size = sum((meta.num_rows if meta.num_rows is not None else 0 for _, meta in blocks_in))
            pool_max_block_size = total_size // self.max_size
            min_rows_per_block = max(min_rows_per_block, pool_max_block_size)
        if min_rows_per_block > 0:
            _check_batch_size(blocks_in, min_rows_per_block, name)
            block_bundles: List[Tuple[Tuple[ObjectRef[Block]], Tuple[BlockMetadata]]] = _bundle_blocks_up_to_size(blocks_in, min_rows_per_block)
        else:
            block_bundles = [((b,), (m,)) for b, m in blocks_in]
        del blocks_in
        owned_by_consumer = block_list._owned_by_consumer
        if clear_input_blocks:
            block_list.clear()
        orig_num_blocks = len(block_bundles)
        results = []
        name = name.title()
        map_bar = ProgressBar(name, total=orig_num_blocks)

        class BlockWorker:

            def __init__(self, *fn_constructor_args: Any, **fn_constructor_kwargs: Any):
                if not isinstance(fn, CallableClass):
                    if fn_constructor_args or fn_constructor_kwargs:
                        raise ValueError(f'fn_constructor_{{kw}}args only valid for CallableClass UDFs, but got: {fn}')
                    self.fn = fn
                else:
                    self.fn = fn(*fn_constructor_args, **fn_constructor_kwargs)

            def ready(self):
                return 'ok'

            def map_block_split(self, input_files: List[str], num_blocks: int, *blocks_and_fn_args, **fn_kwargs) -> BlockPartition:
                return _map_block_split(block_fn, input_files, self.fn, num_blocks, *blocks_and_fn_args, **fn_kwargs)

            @ray.method(num_returns=2)
            def map_block_nosplit(self, input_files: List[str], num_blocks: int, *blocks_and_fn_args, **fn_kwargs) -> Tuple[Block, BlockMetadata]:
                return _map_block_nosplit(block_fn, input_files, self.fn, num_blocks, *blocks_and_fn_args, **fn_kwargs)
        if 'num_cpus' not in remote_args:
            remote_args['num_cpus'] = 1
        if 'scheduling_strategy' not in remote_args:
            ctx = DataContext.get_current()
            remote_args['scheduling_strategy'] = ctx.scheduling_strategy
        BlockWorker = ray.remote(**remote_args)(BlockWorker)
        workers = [BlockWorker.remote(*fn_constructor_args, **fn_constructor_kwargs) for _ in range(self.min_size)]
        tasks = {w.ready.remote(): w for w in workers}
        tasks_in_flight = collections.defaultdict(int)
        max_tasks_in_flight_per_actor = self.max_tasks_in_flight_per_actor or DEFAULT_MAX_TASKS_IN_FLIGHT_PER_ACTOR
        metadata_mapping = {}
        block_indices = {}
        ready_workers = set()
        try:
            while len(results) < orig_num_blocks:
                ready, _ = ray.wait(list(tasks.keys()), timeout=0.01, num_returns=1, fetch_local=False)
                if not ready:
                    if len(workers) < self.max_size and len(ready_workers) / len(workers) > self.ready_to_total_workers_ratio:
                        w = BlockWorker.remote(*fn_constructor_args, **fn_constructor_kwargs)
                        workers.append(w)
                        tasks[w.ready.remote()] = w
                        map_bar.set_description('Map Progress ({} actors {} pending)'.format(len(ready_workers), len(workers) - len(ready_workers)))
                    continue
                [obj_id] = ready
                worker = tasks.pop(obj_id)
                if worker in ready_workers:
                    results.append(obj_id)
                    tasks_in_flight[worker] -= 1
                    map_bar.update(1)
                else:
                    ready_workers.add(worker)
                    map_bar.set_description('Map Progress ({} actors {} pending)'.format(len(ready_workers), len(workers) - len(ready_workers)))
                while block_bundles and tasks_in_flight[worker] < max_tasks_in_flight_per_actor:
                    blocks, metas = block_bundles.pop()
                    ref, meta_ref = worker.map_block_nosplit.remote([f for meta in metas for f in meta.input_files], len(blocks), *blocks + fn_args, **fn_kwargs)
                    metadata_mapping[ref] = meta_ref
                    tasks[ref] = worker
                    block_indices[ref] = len(block_bundles)
                    tasks_in_flight[worker] += 1
            map_bar.close()
            self.num_workers += len(workers)
            new_blocks, new_metadata = ([], [])
            results.sort(key=block_indices.get)
            for block in results:
                new_blocks.append(block)
                new_metadata.append(metadata_mapping[block])
            new_metadata = ray.get(new_metadata)
            return BlockList(new_blocks, new_metadata, owned_by_consumer=owned_by_consumer)
        except Exception as e:
            try:
                for worker in workers:
                    ray.kill(worker)
            except Exception as err:
                logger.exception(f'Error killing workers: {err}')
            finally:
                raise e from None

    def __eq__(self, other: Any) -> bool:
        return isinstance(other, ActorPoolStrategy) and (self.min_size == other.min_size and self.max_size == other.max_size and (self.max_tasks_in_flight_per_actor == other.max_tasks_in_flight_per_actor))