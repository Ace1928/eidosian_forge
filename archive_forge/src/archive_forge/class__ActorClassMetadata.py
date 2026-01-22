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
class _ActorClassMetadata:
    """Metadata for an actor class.

    Attributes:
        language: The actor language, e.g. Python, Java.
        modified_class: The original class that was decorated (with some
            additional methods added like __ray_terminate__).
        actor_creation_function_descriptor: The function descriptor for
            the actor creation task.
        class_id: The ID of this actor class.
        class_name: The name of this class.
        num_cpus: The default number of CPUs required by the actor creation
            task.
        num_gpus: The default number of GPUs required by the actor creation
            task.
        memory: The heap memory quota for this actor.
        resources: The default resources required by the actor creation task.
        accelerator_type: The specified type of accelerator required for the
            node on which this actor runs.
            See :ref:`accelerator types <accelerator_types>`.
        runtime_env: The runtime environment for this actor.
        scheduling_strategy: Strategy about how to schedule this actor.
        last_export_session_and_job: A pair of the last exported session
            and job to help us to know whether this function was exported.
            This is an imperfect mechanism used to determine if we need to
            export the remote function again. It is imperfect in the sense that
            the actor class definition could be exported multiple times by
            different workers.
        method_meta: The actor method metadata.
    """

    def __init__(self, language, modified_class, actor_creation_function_descriptor, class_id, max_restarts, max_task_retries, num_cpus, num_gpus, memory, object_store_memory, resources, accelerator_type, runtime_env, concurrency_groups, scheduling_strategy: SchedulingStrategyT):
        self.language = language
        self.modified_class = modified_class
        self.actor_creation_function_descriptor = actor_creation_function_descriptor
        self.class_name = actor_creation_function_descriptor.class_name
        self.is_cross_language = language != Language.PYTHON
        self.class_id = class_id
        self.max_restarts = max_restarts
        self.max_task_retries = max_task_retries
        self.num_cpus = num_cpus
        self.num_gpus = num_gpus
        self.memory = memory
        self.object_store_memory = object_store_memory
        self.resources = resources
        self.accelerator_type = accelerator_type
        self.runtime_env = runtime_env
        self.concurrency_groups = concurrency_groups
        self.scheduling_strategy = scheduling_strategy
        self.last_export_session_and_job = None
        self.method_meta = _ActorClassMethodMetadata.create(modified_class, actor_creation_function_descriptor)