import math
import threading
import time
from collections import defaultdict, deque
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple
import ray
from ray.data._internal.dataset_logger import DatasetLogger
from ray.data._internal.execution.autoscaling_requester import (
from ray.data._internal.execution.backpressure_policy import BackpressurePolicy
from ray.data._internal.execution.interfaces import (
from ray.data._internal.execution.interfaces.physical_operator import (
from ray.data._internal.execution.operators.base_physical_operator import (
from ray.data._internal.execution.operators.input_data_buffer import InputDataBuffer
from ray.data._internal.execution.util import memory_string
from ray.data._internal.progress_bar import ProgressBar
from ray.data.context import DataContext
class OpState:
    """The execution state tracked for each PhysicalOperator.

    This tracks state to manage input and output buffering for StreamingExecutor and
    progress bars, which is separate from execution state internal to the operators.

    Note: we use the `deque` data structure here because it is thread-safe, enabling
    operator queues to be shared across threads.
    """

    def __init__(self, op: PhysicalOperator, inqueues: List[OpBufferQueue]):
        assert len(inqueues) == len(op.input_dependencies), (op, inqueues)
        self.inqueues: List[OpBufferQueue] = inqueues
        self.outqueue: OpBufferQueue = OpBufferQueue()
        self.op = op
        self.progress_bar = None
        self.num_completed_tasks = 0
        self.inputs_done_called = False
        self.input_done_called = [False] * len(op.input_dependencies)
        self.dependents_completed_called = False
        self._finished: bool = False
        self._exception: Optional[Exception] = None

    def __repr__(self):
        return f'OpState({self.op.name})'

    def initialize_progress_bars(self, index: int, verbose_progress: bool) -> int:
        """Create progress bars at the given index (line offset in console).

        For AllToAllOperator, zero or more sub progress bar would be created.
        Return the number of progress bars created for this operator.
        """
        is_all_to_all = isinstance(self.op, AllToAllOperator)
        enabled = verbose_progress or is_all_to_all
        self.progress_bar = ProgressBar('- ' + self.op.name, self.op.num_outputs_total(), index, enabled=enabled)
        if enabled:
            num_bars = 1
            if is_all_to_all:
                num_bars += self.op.initialize_sub_progress_bars(index + 1)
        else:
            num_bars = 0
        return num_bars

    def close_progress_bars(self):
        """Close all progress bars for this operator."""
        if self.progress_bar:
            self.progress_bar.close()
            if isinstance(self.op, AllToAllOperator):
                self.op.close_sub_progress_bars()

    def num_queued(self) -> int:
        """Return the number of queued bundles across all inqueues."""
        return sum((len(q) for q in self.inqueues))

    def num_processing(self):
        """Return the number of bundles currently in processing for this operator."""
        return self.op.num_active_tasks() + self.op.internal_queue_size()

    def add_output(self, ref: RefBundle) -> None:
        """Move a bundle produced by the operator to its outqueue."""
        self.outqueue.append(ref)
        self.num_completed_tasks += 1
        if self.progress_bar:
            self.progress_bar.update(1, self.op._estimated_output_blocks)

    def refresh_progress_bar(self) -> None:
        """Update the console with the latest operator progress."""
        if self.progress_bar:
            self.progress_bar.set_description(self.summary_str())

    def summary_str(self) -> str:
        queued = self.num_queued() + self.op.internal_queue_size()
        active = self.op.num_active_tasks()
        desc = f'- {self.op.name}: {active} active, {queued} queued'
        mem = memory_string((self.op.current_resource_usage().object_store_memory or 0) + self.inqueue_memory_usage())
        desc += f', {mem} objects'
        suffix = self.op.progress_str()
        if suffix:
            desc += f', {suffix}'
        return desc

    def dispatch_next_task(self) -> None:
        """Move a bundle from the operator inqueue to the operator itself."""
        for i, inqueue in enumerate(self.inqueues):
            ref = inqueue.pop()
            if ref is not None:
                self.op.add_input(ref, input_index=i)
                return
        assert False, 'Nothing to dispatch'

    def get_output_blocking(self, output_split_idx: Optional[int]) -> RefBundle:
        """Get an item from this node's output queue, blocking as needed.

        Returns:
            The RefBundle from the output queue, or an error / end of stream indicator.

        Raises:
            StopIteration: If all outputs are already consumed.
            Exception: If there was an exception raised during execution.
        """
        while True:
            if self._exception is not None:
                raise self._exception
            elif self._finished and (not self.outqueue.has_next(output_split_idx)):
                raise StopIteration()
            ref = self.outqueue.pop(output_split_idx)
            if ref is not None:
                return ref
            time.sleep(0.01)

    def inqueue_memory_usage(self) -> int:
        """Return the object store memory of this operator's inqueue."""
        total = 0
        for op, inq in zip(self.op.input_dependencies, self.inqueues):
            if not isinstance(op, InputDataBuffer):
                total += inq.memory_usage
        return total

    def outqueue_memory_usage(self) -> int:
        """Return the object store memory of this operator's outqueue."""
        return self.outqueue.memory_usage

    def outqueue_num_blocks(self) -> int:
        """Return the number of blocks in this operator's outqueue."""
        return self.outqueue.num_blocks

    def mark_finished(self, exception: Optional[Exception]=None):
        """Marks this operator as finished. Used for exiting get_output_blocking."""
        if exception is None:
            self._finished = True
        else:
            self._exception = exception