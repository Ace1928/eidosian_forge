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
@classmethod
def _sibling(cls) -> type[EnvWithSibilings]:
    """Get a parameter sibling."""
    return RangePartitioningGroupby