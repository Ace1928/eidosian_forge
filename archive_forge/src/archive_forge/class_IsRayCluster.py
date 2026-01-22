import importlib
import os
import secrets
import sys
import warnings
from textwrap import dedent
from typing import Any, Optional
from packaging import version
from pandas.util._decorators import doc  # type: ignore[attr-defined]
from modin.config.pubsub import (
class IsRayCluster(EnvironmentVariable, type=bool):
    """Whether Modin is running on pre-initialized Ray cluster."""
    varname = 'MODIN_RAY_CLUSTER'