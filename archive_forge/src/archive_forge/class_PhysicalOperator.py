from abc import ABC, abstractmethod
from typing import Any, Callable, Dict, List, Optional, Union
import ray
from .ref_bundle import RefBundle
from ray._raylet import ObjectRefGenerator
from ray.data._internal.execution.interfaces.execution_options import (
from ray.data._internal.execution.interfaces.op_runtime_metrics import OpRuntimeMetrics
from ray.data._internal.logical.interfaces import Operator
from ray.data._internal.stats import StatsDict
from ray.data.context import DataContext
class PhysicalOperator(Operator):
    """Abstract class for physical operators.

    An operator transforms one or more input streams of RefBundles into a single
    output stream of RefBundles.

    Physical operators are stateful and non-serializable; they live on the driver side
    of the Dataset only.

    Here's a simple example of implementing a basic "Map" operator:

        class MapOperator(PhysicalOperator):
            def __init__(self):
                self.active_tasks = []

            def add_input(self, refs, _):
                self.active_tasks.append(map_task.remote(refs))

            def has_next(self):
                ready, _ = ray.wait(self.active_tasks, timeout=0)
                return len(ready) > 0

            def get_next(self):
                ready, remaining = ray.wait(self.active_tasks, num_returns=1)
                self.active_tasks = remaining
                return ready[0]

    Note that the above operator fully supports both bulk and streaming execution,
    since `add_input` and `get_next` can be called in any order. In bulk execution,
    all inputs would be added up-front, but in streaming execution the calls could
    be interleaved.
    """

    def __init__(self, name: str, input_dependencies: List['PhysicalOperator'], target_max_block_size: Optional[int]):
        super().__init__(name, input_dependencies)
        for x in input_dependencies:
            assert isinstance(x, PhysicalOperator), x
        self._inputs_complete = not input_dependencies
        self._target_max_block_size = target_max_block_size
        self._dependents_complete = False
        self._started = False
        self._metrics = OpRuntimeMetrics(self)
        self._estimated_output_blocks = None

    def __reduce__(self):
        raise ValueError('Operator is not serializable.')

    @property
    def target_max_block_size(self) -> Optional[int]:
        """
        Target max block size output by this operator. If this returns None,
        then the default from DataContext should be used.
        """
        return self._target_max_block_size

    @property
    def actual_target_max_block_size(self) -> int:
        """
        The actual target max block size output by this operator.
        """
        target_max_block_size = self._target_max_block_size
        if target_max_block_size is None:
            target_max_block_size = DataContext.get_current().target_max_block_size
        return target_max_block_size

    def set_target_max_block_size(self, target_max_block_size: Optional[int]):
        self._target_max_block_size = target_max_block_size

    def completed(self) -> bool:
        """Return True when this operator is completed.

        An operator is completed if any of the following conditions are met:
        - All upstream operators are completed and all outputs are taken.
        - All downstream operators are completed.
        """
        return self._inputs_complete and self.num_active_tasks() == 0 and (not self.has_next()) or self._dependents_complete

    def get_stats(self) -> StatsDict:
        """Return recorded execution stats for use with DatasetStats."""
        raise NotImplementedError

    @property
    def metrics(self) -> OpRuntimeMetrics:
        """Returns the runtime metrics of this operator."""
        self._metrics._extra_metrics = self._extra_metrics()
        return self._metrics

    def _extra_metrics(self) -> Dict[str, Any]:
        """Subclasses should override this method to report extra metrics
        that are specific to them."""
        return {}

    def progress_str(self) -> str:
        """Return any extra status to be displayed in the operator progress bar.

        For example, `<N> actors` to show current number of actors in an actor pool.
        """
        return ''

    def num_outputs_total(self) -> int:
        """Returns the total number of output bundles of this operator.

        The value returned may be an estimate based off the consumption so far.
        This is useful for reporting progress.
        """
        if self._estimated_output_blocks is not None:
            return self._estimated_output_blocks
        if len(self.input_dependencies) == 1:
            return self.input_dependencies[0].num_outputs_total()
        raise AttributeError

    def start(self, options: ExecutionOptions) -> None:
        """Called by the executor when execution starts for an operator.

        Args:
            options: The global options used for the overall execution.
        """
        self._started = True

    def should_add_input(self) -> bool:
        """Return whether it is desirable to add input to this operator right now.

        Operators can customize the implementation of this method to apply additional
        backpressure (e.g., waiting for internal actors to be created).
        """
        return True

    def need_more_inputs(self) -> bool:
        """Return true if the operator still needs more inputs.

        Once this return false, it should never return true again.
        """
        return True

    def add_input(self, refs: RefBundle, input_index: int) -> None:
        """Called when an upstream result is available.

        Inputs may be added in any order, and calls to `add_input` may be interleaved
        with calls to `get_next` / `has_next` to implement streaming execution.

        Subclasses should override `_add_input_inner` instead of this method.

        Args:
            refs: The ref bundle that should be added as input.
            input_index: The index identifying the input dependency producing the
                input. For most operators, this is always `0` since there is only
                one upstream input operator.
        """
        self._metrics.on_input_received(refs)
        self._add_input_inner(refs, input_index)

    def _add_input_inner(self, refs: RefBundle, input_index: int) -> None:
        """Subclasses should override this method to implement `add_input`."""
        raise NotImplementedError

    def input_done(self, input_index: int) -> None:
        """Called when the upstream operator at index `input_index` has completed().

        After this is called, the executor guarantees that no more inputs will be added
        via `add_input` for the given input index.
        """
        pass

    def all_inputs_done(self) -> None:
        """Called when all upstream operators have completed().

        After this is called, the executor guarantees that no more inputs will be added
        via `add_input` for any input index.
        """
        self._inputs_complete = True

    def all_dependents_complete(self) -> None:
        """Called when all downstream operators have completed().

        After this is called, the operator is marked as completed.
        """
        self._dependents_complete = True

    def has_next(self) -> bool:
        """Returns when a downstream output is available.

        When this returns true, it is safe to call `get_next()`.
        """
        raise NotImplementedError

    def get_next(self) -> RefBundle:
        """Get the next downstream output.

        It is only allowed to call this if `has_next()` has returned True.

        Subclasses should override `_get_next_inner` instead of this method.
        """
        output = self._get_next_inner()
        self._metrics.on_output_taken(output)
        return output

    def _get_next_inner(self) -> RefBundle:
        """Subclasses should override this method to implement `get_next`."""
        raise NotImplementedError

    def get_active_tasks(self) -> List[OpTask]:
        """Get a list of the active tasks of this operator."""
        return []

    def num_active_tasks(self) -> int:
        """Return the number of active tasks.

        Subclasses can override this as a performance optimization.
        """
        return len(self.get_active_tasks())

    def throttling_disabled(self) -> bool:
        """Whether to disable resource throttling for this operator.

        This should return True for operators that only manipulate bundle metadata
        (e.g., the OutputSplitter operator). This hints to the execution engine that
        these operators should not be throttled based on resource usage.
        """
        return False

    def internal_queue_size(self) -> int:
        """If the operator has an internal input queue, return its size.

        This is used to report tasks pending submission to actor pools.
        """
        return 0

    def shutdown(self) -> None:
        """Abort execution and release all resources used by this operator.

        This release any Ray resources acquired by this operator such as active
        tasks, actors, and objects.
        """
        if not self._started:
            raise ValueError('Operator must be started before being shutdown.')

    def current_resource_usage(self) -> ExecutionResources:
        """Returns the current estimated resource usage of this operator.

        This method is called by the executor to decide how to allocate resources
        between different operators.
        """
        return ExecutionResources()

    def base_resource_usage(self) -> ExecutionResources:
        """Returns the minimum amount of resources required for execution.

        For example, an operator that creates an actor pool requiring 8 GPUs could
        return ExecutionResources(gpu=8) as its base usage.
        """
        return ExecutionResources()

    def incremental_resource_usage(self) -> ExecutionResources:
        """Returns the incremental resources required for processing another input.

        For example, an operator that launches a task per input could return
        ExecutionResources(cpu=1) as its incremental usage.
        """
        return ExecutionResources()

    def notify_resource_usage(self, input_queue_size: int, under_resource_limits: bool) -> None:
        """Called periodically by the executor.

        Args:
            input_queue_size: The number of inputs queued outside this operator.
            under_resource_limits: Whether this operator is under resource limits.
        """
        pass