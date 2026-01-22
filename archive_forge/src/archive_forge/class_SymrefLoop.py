import os
import warnings
from contextlib import suppress
from typing import Any, Dict, Optional, Set
from .errors import PackedRefsException, RefFormatError
from .file import GitFile, ensure_dir_exists
from .objects import ZERO_SHA, ObjectID, Tag, git_line, valid_hexsha
from .pack import ObjectContainer
class SymrefLoop(Exception):
    """There is a loop between one or more symrefs."""

    def __init__(self, ref, depth) -> None:
        self.ref = ref
        self.depth = depth