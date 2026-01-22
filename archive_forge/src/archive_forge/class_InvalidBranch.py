from __future__ import annotations
import os
import platform
import random
import re
import typing as t
from ..config import (
from ..io import (
from ..git import (
from ..util import (
from . import (
class InvalidBranch(ApplicationError):
    """Exception for invalid branch specification."""

    def __init__(self, branch: str, reason: str) -> None:
        message = 'Invalid branch: %s\n%s' % (branch, reason)
        super().__init__(message)
        self.branch = branch