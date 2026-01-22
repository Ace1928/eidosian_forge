from __future__ import annotations
import ast
import collections
import os
import re
import sys
import token
import tokenize
from dataclasses import dataclass
from types import CodeType
from typing import (
from coverage import env
from coverage.bytecode import code_objects
from coverage.debug import short_stack
from coverage.exceptions import NoSource, NotPython
from coverage.misc import join_regex, nice_pair
from coverage.phystokens import generate_tokens
from coverage.types import TArc, TLineNo
def _process_exits(self, exits: set[ArcStart], add_arc: TAddArcFn, from_set: set[ArcStart] | None=None) -> bool:
    """Helper to process the four kinds of exits."""
    for xit in exits:
        add_arc(xit.lineno, self.start, xit.cause)
    if from_set is not None:
        from_set.update(exits)
    return True