import difflib
from dataclasses import dataclass
from typing import Collection, Iterator, List, Sequence, Set, Tuple, Union
from black.nodes import (
from blib2to3.pgen2.token import ASYNC, NEWLINE
def _find_lines_mapping_index(original_line: int, lines_mappings: Sequence[_LinesMapping], start_index: int) -> int:
    """Returns the original index of the lines mappings for the original line."""
    index = start_index
    while index < len(lines_mappings):
        mapping = lines_mappings[index]
        if mapping.original_start <= original_line <= mapping.original_end:
            return index
        index += 1
    return index