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
class AsvImplementation(EnvironmentVariable, type=ExactStr):
    """Allows to select a library that we will use for testing performance."""
    varname = 'MODIN_ASV_USE_IMPL'
    choices = ('modin', 'pandas')
    default = 'modin'