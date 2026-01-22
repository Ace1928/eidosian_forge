import logging
import os
from pathlib import Path
from types import ModuleType
from typing import Any, Dict, List, Optional
from ray._private.runtime_env.context import RuntimeEnvContext
from ray._private.runtime_env.packaging import (
from ray._private.runtime_env.plugin import RuntimeEnvPlugin
from ray._private.runtime_env.working_dir import set_pythonpath_in_context
from ray._private.utils import get_directory_size_bytes, try_to_create_directory
from ray.exceptions import RuntimeEnvSetupError
def _check_is_uri(s: str) -> bool:
    try:
        protocol, path = parse_uri(s)
    except ValueError:
        protocol, path = (None, None)
    if protocol in Protocol.remote_protocols() and (not path.endswith('.zip')) and (not path.endswith('.whl')):
        raise ValueError('Only .zip or .whl files supported for remote URIs.')
    return protocol is not None