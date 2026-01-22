from tokenize import (generate_tokens, untokenize, TokenError,
from keyword import iskeyword
import ast
import unicodedata
from io import StringIO
import builtins
import types
from typing import Tuple as tTuple, Dict as tDict, Any, Callable, \
from sympy.assumptions.ask import AssumptionKeys
from sympy.core.basic import Basic
from sympy.core import Symbol
from sympy.core.function import Function
from sympy.utilities.misc import func_name
from sympy.functions.elementary.miscellaneous import Max, Min
class AppliedFunction:
    """
    A group of tokens representing a function and its arguments.

    `exponent` is for handling the shorthand sin^2, ln^2, etc.
    """

    def __init__(self, function: TOKEN, args: ParenthesisGroup, exponent=None):
        if exponent is None:
            exponent = []
        self.function = function
        self.args = args
        self.exponent = exponent
        self.items = ['function', 'args', 'exponent']

    def expand(self) -> List[TOKEN]:
        """Return a list of tokens representing the function"""
        return [self.function, *self.args]

    def __getitem__(self, index):
        return getattr(self, self.items[index])

    def __repr__(self):
        return 'AppliedFunction(%s, %s, %s)' % (self.function, self.args, self.exponent)