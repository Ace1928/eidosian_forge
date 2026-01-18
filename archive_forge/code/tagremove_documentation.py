from __future__ import annotations
from traitlets import Set, Unicode
from .base import Preprocessor

        Checks that an output has a tag that indicates removal.

        Returns: Boolean.
        True means output should *not* be removed.
        