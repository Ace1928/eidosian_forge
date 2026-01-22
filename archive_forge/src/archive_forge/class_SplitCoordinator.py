import copy
import logging
import threading
import time
from dataclasses import replace
from typing import TYPE_CHECKING, Dict, Iterator, List, Optional, Tuple, Union
import ray
from ray.data._internal.execution.interfaces import NodeIdStr, RefBundle
from ray.data._internal.execution.legacy_compat import execute_to_legacy_bundle_iterator
from ray.data._internal.execution.operators.output_splitter import OutputSplitter
from ray.data._internal.execution.streaming_executor import StreamingExecutor
from ray.data._internal.stats import DatasetStats, DatasetStatsSummary
from ray.data._internal.util import create_dataset_tag
from ray.data.block import Block, BlockMetadata
from ray.data.iterator import DataIterator
from ray.types import ObjectRef
from ray.util.debug import log_once
from ray.util.scheduling_strategies import NodeAffinitySchedulingStrategy
@ray.remote(num_cpus=0)
class SplitCoordinator:
    """Coordinator actor for routing blocks to output splits.

    This actor runs a streaming executor locally on its main thread. Clients can
    retrieve results via actor calls running on other threads.
    """

    def __init__(self, dataset: 'Dataset', n: int, equal: bool, locality_hints: Optional[List[NodeIdStr]]):
        if locality_hints:
            dataset.context.execution_options.locality_with_output = locality_hints
            logger.info(f'Auto configuring locality_with_output={locality_hints}')
        ray.data.DataContext._set_current(dataset.context)
        self._base_dataset = dataset
        self._n = n
        self._equal = equal
        self._locality_hints = locality_hints
        self._lock = threading.RLock()
        self._executor = None
        self._next_bundle: Dict[int, RefBundle] = {}
        self._unfinished_clients_in_epoch = n
        self._cur_epoch = -1

        def gen_epochs():
            while True:
                executor = StreamingExecutor(copy.deepcopy(dataset.context.execution_options), create_dataset_tag(self._base_dataset._name, self._base_dataset._uuid))
                self._executor = executor

                def add_split_op(dag):
                    return OutputSplitter(dag, n, equal, locality_hints)
                output_iterator = execute_to_legacy_bundle_iterator(executor, dataset._plan, True, dataset._plan._dataset_uuid, dag_rewrite=add_split_op)
                yield output_iterator
        self._next_epoch = gen_epochs()
        self._output_iterator = None

    def stats(self) -> DatasetStatsSummary:
        """Returns stats from the base dataset."""
        if self._executor:
            return self._executor.get_stats().to_summary()
        return self._base_dataset._get_stats_summary()

    def start_epoch(self, split_idx: int) -> str:
        """Called to start an epoch.

        Returns:
            UUID for the epoch, which must be used when accessing results via get().
        """
        epoch_id = self._barrier(split_idx)
        return epoch_id

    def get(self, epoch_id: int, output_split_idx: int) -> Optional[Tuple[ObjectRef[Block], BlockMetadata]]:
        """Blocking get operation.

        This is intended to be called concurrently from multiple clients.
        """
        if epoch_id != self._cur_epoch:
            raise ValueError('Invalid iterator: the dataset has moved on to another epoch.')
        try:
            with self._lock:
                if output_split_idx in self._next_bundle:
                    next_bundle = self._next_bundle[output_split_idx]
                else:
                    next_bundle = None
            while next_bundle is None or not next_bundle.blocks:
                next_bundle = self._output_iterator.get_next(output_split_idx)
            block = next_bundle.blocks[-1]
            next_bundle = replace(next_bundle, blocks=next_bundle.blocks[:-1])
            with self._lock:
                self._next_bundle[output_split_idx] = next_bundle
                if not next_bundle.blocks:
                    del self._next_bundle[output_split_idx]
            return block
        except StopIteration:
            return None

    def _barrier(self, split_idx: int) -> int:
        """Arrive and block until the start of the given epoch."""
        with self._lock:
            starting_epoch = self._cur_epoch
            self._unfinished_clients_in_epoch -= 1
        start_time = time.time()
        while self._cur_epoch == starting_epoch and self._unfinished_clients_in_epoch != 0:
            if time.time() - start_time > BLOCKED_CLIENT_WARN_TIMEOUT:
                if log_once(f'stream_split_blocked_{split_idx}_{starting_epoch}'):
                    logger.warning(f'StreamSplitDataIterator(epoch={starting_epoch}, split={split_idx}) blocked waiting on other clients for more than {BLOCKED_CLIENT_WARN_TIMEOUT}s. All clients must read from the DataIterator splits at the same time. This warning will not be printed again for this epoch.')
            time.sleep(0.1)
        with self._lock:
            if self._cur_epoch == starting_epoch:
                self._cur_epoch += 1
                self._unfinished_clients_in_epoch = self._n
                self._output_iterator = next(self._next_epoch)
        assert self._output_iterator is not None
        return starting_epoch + 1