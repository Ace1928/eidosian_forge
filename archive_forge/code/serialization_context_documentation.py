import contextlib
from typing import List, Any, Dict
from ray.util.serialization import register_serializer, deregister_serializer
from ray.workflow.common import WorkflowRef

    This context only read workflow arguments. Workflows inside
    are untouched and can be serialized again properly.
    