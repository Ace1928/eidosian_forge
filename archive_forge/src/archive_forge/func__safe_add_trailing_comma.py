import re
import sys
from dataclasses import replace
from enum import Enum, auto
from functools import partial, wraps
from typing import Collection, Iterator, List, Optional, Set, Union, cast
from black.brackets import (
from black.comments import FMT_OFF, generate_comments, list_comments
from black.lines import (
from black.mode import Feature, Mode, Preview
from black.nodes import (
from black.numerics import normalize_numeric_literal
from black.strings import (
from black.trans import (
from blib2to3.pgen2 import token
from blib2to3.pytree import Leaf, Node
def _safe_add_trailing_comma(safe: bool, delimiter_priority: int, line: Line) -> Line:
    if safe and delimiter_priority == COMMA_PRIORITY and (line.leaves[-1].type != token.COMMA) and (line.leaves[-1].type != STANDALONE_COMMENT):
        new_comma = Leaf(token.COMMA, ',')
        line.append(new_comma)
    return line