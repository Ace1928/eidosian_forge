import contextlib
from typing import List, Any, Dict
from ray.util.serialization import register_serializer, deregister_serializer
from ray.workflow.common import WorkflowRef
def _resolve_workflow_refs(index: int) -> Any:
    raise ValueError('There is no context for resolving workflow refs.')