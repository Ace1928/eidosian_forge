from __future__ import annotations
import argparse
import collections.abc as c
import dataclasses
import enum
import os
import types
import typing as t
from ..constants import (
from ..util import (
from ..docker_util import (
from ..completion import (
from ..host_configs import (
from ..data import (
class PythonVersionUnspecifiedError(ApplicationError):
    """A Python version was not specified for a context which is unknown, thus the Python version is unknown."""

    def __init__(self, context: str) -> None:
        super().__init__(f'A Python version was not specified for environment `{context}`. Use the `--python` option to specify a Python version.')