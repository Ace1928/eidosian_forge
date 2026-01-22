import inspect
import logging
import weakref
from typing import Any, Dict, List, Optional, Union
import ray._private.ray_constants as ray_constants
import ray._private.signature as signature
import ray._private.worker
import ray._raylet
from ray import ActorClassID, Language, cross_language
from ray._private import ray_option_utils
from ray._private.async_compat import is_async_func
from ray._private.auto_init_hook import wrap_auto_init
from ray._private.client_mode_hook import (
from ray._private.inspect_util import (
from ray._private.ray_option_utils import _warn_if_using_deprecated_placement_group
from ray._private.utils import get_runtime_env_info, parse_runtime_env
from ray._raylet import (
from ray.exceptions import AsyncioActorExit
from ray.util.annotations import DeveloperAPI, PublicAPI
from ray.util.placement_group import _configure_placement_group_based_on_context
from ray.util.scheduling_strategies import (
from ray.util.tracing.tracing_helper import (
@classmethod
def _deserialization_helper(cls, state, outer_object_ref=None):
    """This is defined in order to make pickling work.

        Args:
            state: The serialized state of the actor handle.
            outer_object_ref: The ObjectRef that the serialized actor handle
                was contained in, if any. This is used for counting references
                to the actor handle.

        """
    worker = ray._private.worker.global_worker
    worker.check_connected()
    if hasattr(worker, 'core_worker'):
        return worker.core_worker.deserialize_and_register_actor_handle(state, outer_object_ref)
    else:
        return cls(state['actor_language'], state['actor_id'], state['max_task_retries'], state['method_is_generator'], state['method_decorators'], state['method_signatures'], state['method_num_returns'], state['method_max_retries'], state['method_retry_exceptions'], state['method_generator_backpressure_num_objects'], state['actor_method_cpus'], state['actor_creation_function_descriptor'], worker.current_session_and_job)