import math
import os
import re
import textwrap
from itertools import chain, groupby
from typing import (TYPE_CHECKING, Any, Dict, Generator, Iterable, List, Optional, Set, Tuple,
from docutils import nodes, writers
from docutils.nodes import Element, Text
from docutils.utils import column_width
from sphinx import addnodes
from sphinx.locale import _, admonitionlabels
from sphinx.util.docutils import SphinxTranslator
def writesep(char: str='-', lineno: Optional[int]=None) -> str:
    """Called on the line *before* lineno.
            Called with no *lineno* for the last sep.
            """
    out: List[str] = []
    for colno, width in enumerate(self.measured_widths):
        if lineno is not None and lineno > 0 and (self[lineno, colno] is self[lineno - 1, colno]):
            out.append(' ' * (width + 2))
        else:
            out.append(char * (width + 2))
    head = '+' if out[0][0] == '-' else '|'
    tail = '+' if out[-1][0] == '-' else '|'
    glue = ['+' if left[0] == '-' or right[0] == '-' else '|' for left, right in zip(out, out[1:])]
    glue.append(tail)
    return head + ''.join(chain.from_iterable(zip(out, glue)))