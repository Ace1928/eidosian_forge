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
class CIAWSSecretAccessKey(EnvironmentVariable, type=str):
    """Set to AWS_SECRET_ACCESS_KEY when running mock S3 tests for Modin in GitHub CI."""
    varname = 'AWS_SECRET_ACCESS_KEY'
    default = 'foobar_secret'