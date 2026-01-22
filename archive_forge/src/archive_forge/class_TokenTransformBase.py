import ast
from codeop import CommandCompiler, Compile
import re
import sys
import tokenize
from typing import List, Tuple, Optional, Any
import warnings
from IPython.utils import tokenutil
class TokenTransformBase:
    """Base class for transformations which examine tokens.

    Special syntax should not be transformed when it occurs inside strings or
    comments. This is hard to reliably avoid with regexes. The solution is to
    tokenise the code as Python, and recognise the special syntax in the tokens.

    IPython's special syntax is not valid Python syntax, so tokenising may go
    wrong after the special syntax starts. These classes therefore find and
    transform *one* instance of special syntax at a time into regular Python
    syntax. After each transformation, tokens are regenerated to find the next
    piece of special syntax.

    Subclasses need to implement one class method (find)
    and one regular method (transform).

    The priority attribute can select which transformation to apply if multiple
    transformers match in the same place. Lower numbers have higher priority.
    This allows "%magic?" to be turned into a help call rather than a magic call.
    """
    priority = 10

    def sortby(self):
        return (self.start_line, self.start_col, self.priority)

    def __init__(self, start):
        self.start_line = start[0] - 1
        self.start_col = start[1]

    @classmethod
    def find(cls, tokens_by_line):
        """Find one instance of special syntax in the provided tokens.

        Tokens are grouped into logical lines for convenience,
        so it is easy to e.g. look at the first token of each line.
        *tokens_by_line* is a list of lists of tokenize.TokenInfo objects.

        This should return an instance of its class, pointing to the start
        position it has found, or None if it found no match.
        """
        raise NotImplementedError

    def transform(self, lines: List[str]):
        """Transform one instance of special syntax found by ``find()``

        Takes a list of strings representing physical lines,
        returns a similar list of transformed lines.
        """
        raise NotImplementedError