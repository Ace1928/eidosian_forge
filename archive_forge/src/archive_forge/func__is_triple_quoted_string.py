import itertools
import math
from dataclasses import dataclass, field
from typing import (
from black.brackets import COMMA_PRIORITY, DOT_PRIORITY, BracketTracker
from black.mode import Mode, Preview
from black.nodes import (
from black.strings import str_width
from blib2to3.pgen2 import token
from blib2to3.pytree import Leaf, Node
@property
def _is_triple_quoted_string(self) -> bool:
    """Is the line a triple quoted string?"""
    if not self or self.leaves[0].type != token.STRING:
        return False
    value = self.leaves[0].value
    if value.startswith(('"""', "'''")):
        return True
    if value.startswith(("r'''", 'r"""', "R'''", 'R"""')):
        return True
    return False