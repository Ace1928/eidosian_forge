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
def _handle__Continue(self, node: ast.Continue) -> set[ArcStart]:
    here = self.line_for_node(node)
    continue_start = ArcStart(here, cause="the continue on line {lineno} wasn't executed")
    self.process_continue_exits({continue_start})
    return set()