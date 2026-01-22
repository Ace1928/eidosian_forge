import logging
import warnings
from enum import Enum
from typing import Any, Callable, List, Optional, Union
from ray._private.pydantic_compat import (
from ray._private.utils import import_attr
from ray.serve._private.constants import (
from ray.util.annotations import Deprecated, PublicAPI
@Deprecated
class DeploymentMode(str, Enum):
    NoServer = 'NoServer'
    HeadOnly = 'HeadOnly'
    EveryNode = 'EveryNode'