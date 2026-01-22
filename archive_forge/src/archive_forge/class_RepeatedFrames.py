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
class RepeatedFrames:
    """
    A sequence of consecutive stack frames which shouldn't be displayed because
    the same code and line number were repeated many times in the stack, e.g.
    because of deep recursion.

    Attributes:
        - frames: list of raw frame or traceback objects
        - frame_keys: list of tuples (frame.f_code, lineno) extracted from the frame objects.
                        It's this information from the frames that is used to determine
                        whether two frames should be considered similar (i.e. repeating).
        - description: A string briefly describing frame_keys
    """

    def __init__(self, frames: List[Union[FrameType, TracebackType]], frame_keys: List[Tuple[CodeType, int]]):
        self.frames = frames
        self.frame_keys = frame_keys

    @cached_property
    def description(self) -> str:
        """
        A string briefly describing the repeated frames, e.g.
            my_function at line 10 (100 times)
        """
        counts = sorted(Counter(self.frame_keys).items(), key=lambda item: (-item[1], item[0][0].co_name))
        return ', '.join(('{name} at line {lineno} ({count} times)'.format(name=Source.for_filename(code.co_filename).code_qualname(code), lineno=lineno, count=count) for (code, lineno), count in counts))

    def __repr__(self):
        return '<{self.__class__.__name__} {self.description}>'.format(self=self)