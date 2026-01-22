from __future__ import annotations
import collections.abc as c
import os
import typing as t
from .....io import (
from .....util import (
from .. import (
class CoverageAnalyzeTargetsConfig(CoverageAnalyzeConfig):
    """Configuration for the `coverage analyze targets` command."""