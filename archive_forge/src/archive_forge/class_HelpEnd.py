import ast
from codeop import CommandCompiler, Compile
import re
import sys
import tokenize
from typing import List, Tuple, Optional, Any
import warnings
from IPython.utils import tokenutil
class HelpEnd(TokenTransformBase):
    """Transformer for help syntax: obj? and obj??"""
    priority = 5

    def __init__(self, start, q_locn):
        super().__init__(start)
        self.q_line = q_locn[0] - 1
        self.q_col = q_locn[1]

    @classmethod
    def find(cls, tokens_by_line):
        """Find the first help command (foo?) in the cell.
        """
        for line in tokens_by_line:
            if len(line) > 2 and line[-2].string == '?':
                ix = 0
                while line[ix].type in {tokenize.INDENT, tokenize.DEDENT}:
                    ix += 1
                return cls(line[ix].start, line[-2].start)

    def transform(self, lines):
        """Transform a help command found by the ``find()`` classmethod.
        """
        piece = ''.join(lines[self.start_line:self.q_line + 1])
        indent, content = (piece[:self.start_col], piece[self.start_col:])
        lines_before = lines[:self.start_line]
        lines_after = lines[self.q_line + 1:]
        m = _help_end_re.search(content)
        if not m:
            raise SyntaxError(content)
        assert m is not None, content
        target = m.group(1)
        esc = m.group(3)
        call = _make_help_call(target, esc)
        new_line = indent + call + '\n'
        return lines_before + [new_line] + lines_after