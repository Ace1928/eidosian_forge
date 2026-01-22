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
def _analyze_ast(self) -> None:
    """Run the AstArcAnalyzer and save its results.

        `_all_arcs` is the set of arcs in the code.

        """
    aaa = AstArcAnalyzer(self.text, self.raw_statements, self._multiline)
    aaa.analyze()
    self._all_arcs = set()
    for l1, l2 in aaa.arcs:
        fl1 = self.first_line(l1)
        fl2 = self.first_line(l2)
        if fl1 != fl2:
            self._all_arcs.add((fl1, fl2))
    self._missing_arc_fragments = aaa.missing_arc_fragments