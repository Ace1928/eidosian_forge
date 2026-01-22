import difflib
from dataclasses import dataclass
from typing import Collection, Iterator, List, Sequence, Set, Tuple, Union
from black.nodes import (
from blib2to3.pgen2.token import ASYNC, NEWLINE
def adjusted_lines(lines: Collection[Tuple[int, int]], original_source: str, modified_source: str) -> List[Tuple[int, int]]:
    """Returns the adjusted line ranges based on edits from the original code.

    This computes the new line ranges by diffing original_source and
    modified_source, and adjust each range based on how the range overlaps with
    the diffs.

    Note the diff can contain lines outside of the original line ranges. This can
    happen when the formatting has to be done in adjacent to maintain consistent
    local results. For example:

    1. def my_func(arg1, arg2,
    2.             arg3,):
    3.   pass

    If it restricts to line 2-2, it can't simply reformat line 2, it also has
    to reformat line 1:

    1. def my_func(
    2.     arg1,
    3.     arg2,
    4.     arg3,
    5. ):
    6.   pass

    In this case, we will expand the line ranges to also include the whole diff
    block.

    Args:
      lines: a collection of line ranges.
      original_source: the original source.
      modified_source: the modified source.
    """
    lines_mappings = _calculate_lines_mappings(original_source, modified_source)
    new_lines = []
    current_mapping_index = 0
    for start, end in sorted(lines):
        start_mapping_index = _find_lines_mapping_index(start, lines_mappings, current_mapping_index)
        end_mapping_index = _find_lines_mapping_index(end, lines_mappings, start_mapping_index)
        current_mapping_index = start_mapping_index
        if start_mapping_index >= len(lines_mappings) or end_mapping_index >= len(lines_mappings):
            continue
        start_mapping = lines_mappings[start_mapping_index]
        end_mapping = lines_mappings[end_mapping_index]
        if start_mapping.is_changed_block:
            new_start = start_mapping.modified_start
        else:
            new_start = start - start_mapping.original_start + start_mapping.modified_start
        if end_mapping.is_changed_block:
            new_end = end_mapping.modified_end
        else:
            new_end = end - end_mapping.original_start + end_mapping.modified_start
        new_range = (new_start, new_end)
        if is_valid_line_range(new_range):
            new_lines.append(new_range)
    return new_lines