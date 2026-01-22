from __future__ import annotations
import base64
import dataclasses
import json
import os
import re
import typing as t
from .encoding import (
from .io import (
from .util import (
from .util_common import (
from .config import (
from .data import (
from .host_configs import (
from .connections import (
from .coverage_util import (
@dataclasses.dataclass(frozen=True)
class PipUninstall(PipCommand):
    """Details required to perform a pip uninstall."""
    packages: list[str]
    ignore_errors: bool