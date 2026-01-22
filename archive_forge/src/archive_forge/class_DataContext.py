import os
import threading
from typing import TYPE_CHECKING, Any, Dict, List, Optional
import ray
from ray._private.ray_constants import env_integer
from ray.util.annotations import DeveloperAPI
from ray.util.scheduling_strategies import SchedulingStrategyT
@DeveloperAPI
class DataContext:
    """Singleton for shared Dataset resources and configurations.

    This object is automatically propagated to workers and can be retrieved
    from the driver and remote workers via DataContext.get_current().
    """

    def __init__(self, target_max_block_size: int, target_shuffle_max_block_size: int, target_min_block_size: int, streaming_read_buffer_size: int, enable_pandas_block: bool, optimize_fuse_stages: bool, optimize_fuse_read_stages: bool, optimize_fuse_shuffle_stages: bool, optimize_reorder_stages: bool, actor_prefetcher_enabled: bool, use_push_based_shuffle: bool, pipeline_push_based_shuffle_reduce_tasks: bool, scheduling_strategy: SchedulingStrategyT, scheduling_strategy_large_args: SchedulingStrategyT, large_args_threshold: int, use_polars: bool, new_execution_backend: bool, use_streaming_executor: bool, eager_free: bool, decoding_size_estimation: bool, min_parallelism: bool, enable_tensor_extension_casting: bool, enable_auto_log_stats: bool, trace_allocations: bool, optimizer_enabled: bool, execution_options: 'ExecutionOptions', use_ray_tqdm: bool, enable_progress_bars: bool, enable_get_object_locations_for_metrics: bool, use_runtime_metrics_scheduling: bool, write_file_retry_on_errors: List[str]):
        """Private constructor (use get_current() instead)."""
        self.target_max_block_size = target_max_block_size
        self.target_shuffle_max_block_size = target_shuffle_max_block_size
        self.target_min_block_size = target_min_block_size
        self.streaming_read_buffer_size = streaming_read_buffer_size
        self.enable_pandas_block = enable_pandas_block
        self.optimize_fuse_stages = optimize_fuse_stages
        self.optimize_fuse_read_stages = optimize_fuse_read_stages
        self.optimize_fuse_shuffle_stages = optimize_fuse_shuffle_stages
        self.optimize_reorder_stages = optimize_reorder_stages
        self.actor_prefetcher_enabled = actor_prefetcher_enabled
        self.use_push_based_shuffle = use_push_based_shuffle
        self.pipeline_push_based_shuffle_reduce_tasks = pipeline_push_based_shuffle_reduce_tasks
        self.scheduling_strategy = scheduling_strategy
        self.scheduling_strategy_large_args = scheduling_strategy_large_args
        self.large_args_threshold = large_args_threshold
        self.use_polars = use_polars
        self.new_execution_backend = new_execution_backend
        self.use_streaming_executor = use_streaming_executor
        self.eager_free = eager_free
        self.decoding_size_estimation = decoding_size_estimation
        self.min_parallelism = min_parallelism
        self.enable_tensor_extension_casting = enable_tensor_extension_casting
        self.enable_auto_log_stats = enable_auto_log_stats
        self.trace_allocations = trace_allocations
        self.optimizer_enabled = optimizer_enabled
        self.execution_options = execution_options
        self.use_ray_tqdm = use_ray_tqdm
        self.enable_progress_bars = enable_progress_bars
        self.enable_get_object_locations_for_metrics = enable_get_object_locations_for_metrics
        self.use_runtime_metrics_scheduling = use_runtime_metrics_scheduling
        self.write_file_retry_on_errors = write_file_retry_on_errors
        self._task_pool_data_task_remote_args: Dict[str, Any] = {}
        self.max_errored_blocks = 0
        self._kv_configs: Dict[str, Any] = {}

    @staticmethod
    def get_current() -> 'DataContext':
        """Get or create a singleton context.

        If the context has not yet been created in this process, it will be
        initialized with default settings.
        """
        global _default_context
        with _context_lock:
            if _default_context is None:
                _default_context = DataContext(target_max_block_size=DEFAULT_TARGET_MAX_BLOCK_SIZE, target_shuffle_max_block_size=DEFAULT_SHUFFLE_TARGET_MAX_BLOCK_SIZE, target_min_block_size=DEFAULT_TARGET_MIN_BLOCK_SIZE, streaming_read_buffer_size=DEFAULT_STREAMING_READ_BUFFER_SIZE, enable_pandas_block=DEFAULT_ENABLE_PANDAS_BLOCK, optimize_fuse_stages=DEFAULT_OPTIMIZE_FUSE_STAGES, optimize_fuse_read_stages=DEFAULT_OPTIMIZE_FUSE_READ_STAGES, optimize_fuse_shuffle_stages=DEFAULT_OPTIMIZE_FUSE_SHUFFLE_STAGES, optimize_reorder_stages=DEFAULT_OPTIMIZE_REORDER_STAGES, actor_prefetcher_enabled=DEFAULT_ACTOR_PREFETCHER_ENABLED, use_push_based_shuffle=DEFAULT_USE_PUSH_BASED_SHUFFLE, pipeline_push_based_shuffle_reduce_tasks=True, scheduling_strategy=DEFAULT_SCHEDULING_STRATEGY, scheduling_strategy_large_args=DEFAULT_SCHEDULING_STRATEGY_LARGE_ARGS, large_args_threshold=DEFAULT_LARGE_ARGS_THRESHOLD, use_polars=DEFAULT_USE_POLARS, new_execution_backend=DEFAULT_NEW_EXECUTION_BACKEND, use_streaming_executor=DEFAULT_USE_STREAMING_EXECUTOR, eager_free=DEFAULT_EAGER_FREE, decoding_size_estimation=DEFAULT_DECODING_SIZE_ESTIMATION_ENABLED, min_parallelism=DEFAULT_MIN_PARALLELISM, enable_tensor_extension_casting=DEFAULT_ENABLE_TENSOR_EXTENSION_CASTING, enable_auto_log_stats=DEFAULT_AUTO_LOG_STATS, trace_allocations=DEFAULT_TRACE_ALLOCATIONS, optimizer_enabled=DEFAULT_OPTIMIZER_ENABLED, execution_options=ray.data.ExecutionOptions(), use_ray_tqdm=DEFAULT_USE_RAY_TQDM, enable_progress_bars=DEFAULT_ENABLE_PROGRESS_BARS, enable_get_object_locations_for_metrics=DEFAULT_ENABLE_GET_OBJECT_LOCATIONS_FOR_METRICS, use_runtime_metrics_scheduling=DEFAULT_USE_RUNTIME_METRICS_SCHEDULING, write_file_retry_on_errors=DEFAULT_WRITE_FILE_RETRY_ON_ERRORS)
            return _default_context

    @staticmethod
    def _set_current(context: 'DataContext') -> None:
        """Set the current context in a remote worker.

        This is used internally by Dataset to propagate the driver context to
        remote workers used for parallelization.
        """
        global _default_context
        _default_context = context

    def get_config(self, key: str, default: Any=None) -> Any:
        """Get the value for a key-value style config.

        Args:
            key: The key of the config.
            default: The default value to return if the key is not found.
        Returns: The value for the key, or the default value if the key is not found.
        """
        return self._kv_configs.get(key, default)

    def set_config(self, key: str, value: Any) -> None:
        """Set the value for a key-value style config.

        Args:
            key: The key of the config.
            value: The value of the config.
        """
        self._kv_configs[key] = value

    def remove_config(self, key: str) -> None:
        """Remove a key-value style config.

        Args:
            key: The key of the config.
        """
        self._kv_configs.pop(key, None)