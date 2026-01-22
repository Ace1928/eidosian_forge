import difflib
from dataclasses import dataclass
from typing import Collection, Iterator, List, Sequence, Set, Tuple, Union
from black.nodes import (
from blib2to3.pgen2.token import ASYNC, NEWLINE
@dataclass
class _LinesMapping:
    """1-based lines mapping from original source to modified source.

    Lines [original_start, original_end] from original source
    are mapped to [modified_start, modified_end].

    The ranges are inclusive on both ends.
    """
    original_start: int
    original_end: int
    modified_start: int
    modified_end: int
    is_changed_block: bool