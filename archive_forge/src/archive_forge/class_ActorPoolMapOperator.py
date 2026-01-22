import collections
from dataclasses import dataclass
from typing import Any, Dict, Iterator, List, Optional, Union
import ray
from ray.data._internal.compute import ActorPoolStrategy
from ray.data._internal.dataset_logger import DatasetLogger
from ray.data._internal.execution.interfaces import (
from ray.data._internal.execution.operators.map_operator import MapOperator, _map_task
from ray.data._internal.execution.operators.map_transformer import MapTransformer
from ray.data._internal.execution.util import locality_string
from ray.data.block import Block, BlockMetadata
from ray.data.context import DataContext
from ray.types import ObjectRef
class ActorPoolMapOperator(MapOperator):
    """A MapOperator implementation that executes tasks on an actor pool.

    This class manages the state of a pool of actors used for task execution, as well
    as dispatch of tasks to those actors.

    It operates in two modes. In bulk mode, tasks are queued internally and executed
    when the operator has free actor slots. In streaming mode, the streaming executor
    only adds input when `should_add_input() = True` (i.e., there are free slots).
    This allows for better control of backpressure (e.g., suppose we go over memory
    limits after adding put, then there isn't any way to "take back" the inputs prior
    to actual execution).
    """

    def __init__(self, map_transformer: MapTransformer, input_op: PhysicalOperator, target_max_block_size: Optional[int], autoscaling_policy: 'AutoscalingPolicy', name: str='ActorPoolMap', min_rows_per_bundle: Optional[int]=None, ray_remote_args: Optional[Dict[str, Any]]=None):
        """Create an ActorPoolMapOperator instance.

        Args:
            transform_fn: The function to apply to each ref bundle input.
            init_fn: The callable class to instantiate on each actor.
            input_op: Operator generating input data for this op.
            autoscaling_policy: A policy controlling when the actor pool should be
                scaled up and scaled down.
            name: The name of this operator.
            target_max_block_size: The target maximum number of bytes to
                include in an output block.
            min_rows_per_bundle: The number of rows to gather per batch passed to the
                transform_fn, or None to use the block size. Setting the batch size is
                important for the performance of GPU-accelerated transform functions.
                The actual rows passed may be less if the dataset is small.
            ray_remote_args: Customize the ray remote args for this op's tasks.
        """
        super().__init__(map_transformer, input_op, name, target_max_block_size, min_rows_per_bundle, ray_remote_args)
        self._ray_remote_args = self._apply_default_remote_args(self._ray_remote_args)
        self._min_rows_per_bundle = min_rows_per_bundle
        self._autoscaling_policy = autoscaling_policy
        self._actor_pool = _ActorPool(autoscaling_policy._config.max_tasks_in_flight)
        self._bundle_queue = collections.deque()
        self._cls = None
        self._inputs_done = False

    def internal_queue_size(self) -> int:
        return len(self._bundle_queue)

    def start(self, options: ExecutionOptions):
        self._actor_locality_enabled = options.actor_locality_enabled
        super().start(options)
        self._cls = ray.remote(**self._ray_remote_args)(_MapWorker)
        for _ in range(self._autoscaling_policy.min_workers):
            self._start_actor()
        refs = self._actor_pool.get_pending_actor_refs()
        logger.get_logger().info(f'{self._name}: Waiting for {len(refs)} pool actors to start...')
        try:
            ray.get(refs, timeout=DEFAULT_WAIT_FOR_MIN_ACTORS_SEC)
        except ray.exceptions.GetTimeoutError:
            raise ray.exceptions.GetTimeoutError('Timed out while starting actors. This may mean that the cluster does not have enough resources for the requested actor pool.')

    def should_add_input(self) -> bool:
        return self._actor_pool.num_free_slots() > 0

    def notify_resource_usage(self, input_queue_size: int, under_resource_limits: bool) -> None:
        free_slots = self._actor_pool.num_free_slots()
        if input_queue_size > free_slots and under_resource_limits:
            self._scale_up_if_needed()
        else:
            self._scale_down_if_needed()

    def _start_actor(self):
        """Start a new actor and add it to the actor pool as a pending actor."""
        assert self._cls is not None
        ctx = DataContext.get_current()
        actor = self._cls.remote(ctx, src_fn_name=self.name, map_transformer=self._map_transformer)
        res_ref = actor.get_location.remote()

        def _task_done_callback(res_ref):
            has_actor = self._actor_pool.pending_to_running(res_ref)
            if not has_actor:
                return
            self._dispatch_tasks()
        self._submit_metadata_task(res_ref, lambda: _task_done_callback(res_ref))
        self._actor_pool.add_pending_actor(actor, res_ref)

    def _add_bundled_input(self, bundle: RefBundle):
        self._bundle_queue.append(bundle)
        self._dispatch_tasks()

    def _dispatch_tasks(self):
        """Try to dispatch tasks from the bundle buffer to the actor pool.

        This is called when:
            * a new input bundle is added,
            * a task finishes,
            * a new worker has been created.
        """
        while self._bundle_queue:
            if self._actor_locality_enabled:
                actor = self._actor_pool.pick_actor(self._bundle_queue[0])
            else:
                actor = self._actor_pool.pick_actor()
            if actor is None:
                break
            bundle = self._bundle_queue.popleft()
            input_blocks = [block for block, _ in bundle.blocks]
            ctx = TaskContext(task_idx=self._next_data_task_idx, target_max_block_size=self.actual_target_max_block_size)
            gen = actor.submit.options(num_returns='streaming', name=self.name).remote(DataContext.get_current(), ctx, *input_blocks)

            def _task_done_callback(actor_to_return):
                self._actor_pool.return_actor(actor_to_return)
                self._dispatch_tasks()
            actor_to_return = actor
            self._submit_data_task(gen, bundle, lambda: _task_done_callback(actor_to_return))
        if self._bundle_queue:
            self._scale_up_if_needed()
        else:
            self._scale_down_if_needed()

    def _scale_up_if_needed(self):
        """Try to scale up the pool if the autoscaling policy allows it."""
        while self._autoscaling_policy.should_scale_up(num_total_workers=self._actor_pool.num_total_actors(), num_running_workers=self._actor_pool.num_running_actors()):
            self._start_actor()

    def _scale_down_if_needed(self):
        """Try to scale down the pool if the autoscaling policy allows it."""
        self._kill_inactive_workers_if_done()
        while self._autoscaling_policy.should_scale_down(num_total_workers=self._actor_pool.num_total_actors(), num_idle_workers=self._actor_pool.num_idle_actors()):
            killed = self._actor_pool.kill_inactive_actor()
            if not killed:
                break

    def all_inputs_done(self):
        super().all_inputs_done()
        self._inputs_done = True
        self._scale_down_if_needed()

    def _kill_inactive_workers_if_done(self):
        if self._inputs_done and (not self._bundle_queue):
            self._actor_pool.kill_all_inactive_actors()

    def shutdown(self):
        self._actor_pool.kill_all_actors()
        super().shutdown()
        min_workers = self._autoscaling_policy.min_workers
        if len(self._output_metadata) < min_workers:
            logger.get_logger().warning(f'To ensure full parallelization across an actor pool of size {min_workers}, the Dataset should consist of at least {min_workers} distinct blocks. Consider increasing the parallelism when creating the Dataset.')

    def progress_str(self) -> str:
        base = f'{self._actor_pool.num_running_actors()} actors'
        pending = self._actor_pool.num_pending_actors()
        if pending:
            base += f' ({pending} pending)'
        if self._actor_locality_enabled:
            base += ' ' + locality_string(self._actor_pool._locality_hits, self._actor_pool._locality_misses)
        else:
            base += ' [locality off]'
        return base

    def base_resource_usage(self) -> ExecutionResources:
        min_workers = self._autoscaling_policy.min_workers
        return ExecutionResources(cpu=self._ray_remote_args.get('num_cpus', 0) * min_workers, gpu=self._ray_remote_args.get('num_gpus', 0) * min_workers)

    def current_resource_usage(self) -> ExecutionResources:
        num_active_workers = self._actor_pool.num_total_actors()
        return ExecutionResources(cpu=self._ray_remote_args.get('num_cpus', 0) * num_active_workers, gpu=self._ray_remote_args.get('num_gpus', 0) * num_active_workers, object_store_memory=self.metrics.obj_store_mem_cur)

    def incremental_resource_usage(self) -> ExecutionResources:
        if self._autoscaling_policy.should_scale_up(num_total_workers=self._actor_pool.num_total_actors(), num_running_workers=self._actor_pool.num_running_actors()):
            num_cpus = self._ray_remote_args.get('num_cpus', 0)
            num_gpus = self._ray_remote_args.get('num_gpus', 0)
        else:
            num_cpus = 0
            num_gpus = 0
        return ExecutionResources(cpu=num_cpus, gpu=num_gpus, object_store_memory=self._metrics.average_bytes_outputs_per_task)

    def _extra_metrics(self) -> Dict[str, Any]:
        res = {}
        if self._actor_locality_enabled:
            res['locality_hits'] = self._actor_pool._locality_hits
            res['locality_misses'] = self._actor_pool._locality_misses
        return res

    @staticmethod
    def _apply_default_remote_args(ray_remote_args: Dict[str, Any]) -> Dict[str, Any]:
        """Apply defaults to the actor creation remote args."""
        ray_remote_args = ray_remote_args.copy()
        if 'scheduling_strategy' not in ray_remote_args:
            ctx = DataContext.get_current()
            ray_remote_args['scheduling_strategy'] = ctx.scheduling_strategy
        if 'max_restarts' not in ray_remote_args:
            ray_remote_args['max_restarts'] = -1
        if 'max_task_retries' not in ray_remote_args and ray_remote_args.get('max_restarts') != 0:
            ray_remote_args['max_task_retries'] = -1
        return ray_remote_args