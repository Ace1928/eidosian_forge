import json
import logging
import os
from copy import deepcopy
from dataclasses import asdict, is_dataclass
from typing import Any, Callable, Dict, List, Optional, Set, Tuple, Union
import ray
from ray._private.ray_constants import DEFAULT_RUNTIME_ENV_TIMEOUT_SECONDS
from ray._private.runtime_env.conda import get_uri as get_conda_uri
from ray._private.runtime_env.pip import get_uri as get_pip_uri
from ray._private.runtime_env.plugin_schema_manager import RuntimeEnvPluginSchemaManager
from ray._private.runtime_env.validation import OPTION_TO_VALIDATION_FN
from ray._private.thirdparty.dacite import from_dict
from ray.core.generated.runtime_env_common_pb2 import (
from ray.util.annotations import PublicAPI
def _merge_runtime_env(parent: Optional[RuntimeEnv], child: Optional[RuntimeEnv], override: bool=False) -> Optional[RuntimeEnv]:
    """Merge the parent and child runtime environments.

    If override = True, the child's runtime env overrides the parent's
    runtime env in the event of a conflict.

    Merging happens per key (i.e., "conda", "pip", ...), but
    "env_vars" are merged per env var key.

    It returns None if Ray fails to merge runtime environments because
    of a conflict and `override = False`.

    Args:
        parent: Parent runtime env.
        child: Child runtime env.
        override: If True, the child's runtime env overrides
            conflicting fields.
    Returns:
        The merged runtime env's if Ray successfully merges them.
        None if the runtime env's conflict. Empty dict if
        parent and child are both None.
    """
    if parent is None:
        parent = {}
    if child is None:
        child = {}
    parent = deepcopy(parent)
    child = deepcopy(child)
    parent_env_vars = parent.pop('env_vars', {})
    child_env_vars = child.pop('env_vars', {})
    if not override:
        if set(parent.keys()).intersection(set(child.keys())):
            return None
        if set(parent_env_vars.keys()).intersection(set(child_env_vars.keys())):
            return None
    parent.update(child)
    parent_env_vars.update(child_env_vars)
    if parent_env_vars:
        parent['env_vars'] = parent_env_vars
    return parent