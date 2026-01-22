from __future__ import annotations
import typing as t
from .io import (
from .util import (
from .ci import (
from .classification import (
from .config import (
from .metadata import (
from .provisioning import (
class AllTargetsSkipped(ApplicationWarning):
    """All targets skipped."""

    def __init__(self) -> None:
        super().__init__('All targets skipped.')