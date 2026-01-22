import uuid
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Union
from ray.util.annotations import PublicAPI
def _get_proto_runtime_env_config(self) -> str:
    """Return the JSON-serialized parsed runtime env info"""
    return self._get_proto_job_config().runtime_env_info.runtime_env_config