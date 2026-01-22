import ast
import html
import os
import sys
from collections import defaultdict, Counter
from enum import Enum
from textwrap import dedent
from types import FrameType, CodeType, TracebackType
from typing import (
from typing import Mapping
import executing
from asttokens.util import Token
from executing import only
from pure_eval import Evaluator, is_expression_interesting
from stack_data.utils import (
def _clean_pieces(self) -> Iterator[range]:
    pieces = self._raw_split_into_pieces(self.tree, 1, len(self.lines) + 1)
    pieces = [(start, end) for start, end in pieces if end > start]
    new_pieces = pieces[:1]
    for start, end in pieces[1:]:
        last_start, last_end = new_pieces[-1]
        if start < last_end:
            assert start == last_end - 1
            assert ';' in self.lines[start - 1]
            new_pieces[-1] = (last_start, end)
        else:
            new_pieces.append((start, end))
    pieces = new_pieces
    starts = [start for start, end in pieces[1:]]
    ends = [end for start, end in pieces[:-1]]
    if starts != ends:
        joins = list(map(set, zip(starts, ends)))
        mismatches = [s for s in joins if len(s) > 1]
        raise AssertionError('Pieces mismatches: %s' % mismatches)

    def is_blank(i):
        try:
            return not self.lines[i - 1].strip()
        except IndexError:
            return False
    for start, end in pieces:
        while is_blank(start):
            start += 1
        while is_blank(end - 1):
            end -= 1
        if start < end:
            yield range(start, end)