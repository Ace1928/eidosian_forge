import uuid
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Union
from ray.util.annotations import PublicAPI
def _runtime_env_has_working_dir(self):
    return self._validate_runtime_env().has_working_dir()