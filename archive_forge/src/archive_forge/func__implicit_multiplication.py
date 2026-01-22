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
def _implicit_multiplication(tokens: List[tUnion[TOKEN, AppliedFunction]], local_dict: DICT, global_dict: DICT):
    """Implicitly adds '*' tokens.

    Cases:

    - Two AppliedFunctions next to each other ("sin(x)cos(x)")

    - AppliedFunction next to an open parenthesis ("sin x (cos x + 1)")

    - A close parenthesis next to an AppliedFunction ("(x+2)sin x")
    - A close parenthesis next to an open parenthesis ("(x+2)(x+3)")

    - AppliedFunction next to an implicitly applied function ("sin(x)cos x")

    """
    result: List[tUnion[TOKEN, AppliedFunction]] = []
    skip = False
    for tok, nextTok in zip(tokens, tokens[1:]):
        result.append(tok)
        if skip:
            skip = False
            continue
        if tok[0] == OP and tok[1] == '.' and (nextTok[0] == NAME):
            skip = True
            continue
        if isinstance(tok, AppliedFunction):
            if isinstance(nextTok, AppliedFunction):
                result.append((OP, '*'))
            elif nextTok == (OP, '('):
                if tok.function[1] == 'Function':
                    tok.function = (tok.function[0], 'Symbol')
                result.append((OP, '*'))
            elif nextTok[0] == NAME:
                result.append((OP, '*'))
        elif tok == (OP, ')'):
            if isinstance(nextTok, AppliedFunction):
                result.append((OP, '*'))
            elif nextTok[0] == NAME:
                result.append((OP, '*'))
            elif nextTok == (OP, '('):
                result.append((OP, '*'))
        elif tok[0] == NAME and (not _token_callable(tok, local_dict, global_dict)):
            if isinstance(nextTok, AppliedFunction) or (nextTok[0] == NAME and _token_callable(nextTok, local_dict, global_dict)):
                result.append((OP, '*'))
            elif nextTok == (OP, '('):
                result.append((OP, '*'))
            elif nextTok[0] == NAME:
                result.append((OP, '*'))
    if tokens:
        result.append(tokens[-1])
    return result