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
def _implicit_application(tokens: List[tUnion[TOKEN, AppliedFunction]], local_dict: DICT, global_dict: DICT):
    """Adds parentheses as needed after functions."""
    result: List[tUnion[TOKEN, AppliedFunction]] = []
    appendParen = 0
    skip = 0
    exponentSkip = False
    for tok, nextTok in zip(tokens, tokens[1:]):
        result.append(tok)
        if tok[0] == NAME and nextTok[0] not in [OP, ENDMARKER, NEWLINE]:
            if _token_callable(tok, local_dict, global_dict, nextTok):
                result.append((OP, '('))
                appendParen += 1
        elif tok[0] == NAME and nextTok[0] == OP and (nextTok[1] == '**'):
            if _token_callable(tok, local_dict, global_dict):
                exponentSkip = True
        elif exponentSkip:
            if isinstance(tok, AppliedFunction) or (tok[0] == OP and tok[1] == '*'):
                if not (nextTok[0] == OP and nextTok[1] == '*'):
                    if not (nextTok[0] == OP and nextTok[1] == '('):
                        result.append((OP, '('))
                        appendParen += 1
                    exponentSkip = False
        elif appendParen:
            if nextTok[0] == OP and nextTok[1] in ('^', '**', '*'):
                skip = 1
                continue
            if skip:
                skip -= 1
                continue
            result.append((OP, ')'))
            appendParen -= 1
    if tokens:
        result.append(tokens[-1])
    if appendParen:
        result.extend([(OP, ')')] * appendParen)
    return result