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
def working_dir_uri(self) -> Optional[str]:
    return self.get('working_dir')