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
class TestDatasetSize(EnvironmentVariable, type=str):
    """Dataset size for running some tests."""
    varname = 'MODIN_TEST_DATASET_SIZE'
    choices = ('Small', 'Normal', 'Big')