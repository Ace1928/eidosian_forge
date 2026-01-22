import ast
import gyp.common
import gyp.simple_copy
import multiprocessing
import os.path
import re
import shlex
import signal
import subprocess
import sys
import threading
import traceback
from distutils.version import StrictVersion
from gyp.common import GypError
from gyp.common import OrderedSet
def EvalSingleCondition(cond_expr, true_dict, false_dict, phase, variables, build_file):
    """Returns true_dict if cond_expr evaluates to true, and false_dict
  otherwise."""
    cond_expr_expanded = ExpandVariables(cond_expr, phase, variables, build_file)
    if type(cond_expr_expanded) not in (str, int):
        raise ValueError('Variable expansion in this context permits str and int ' + 'only, found ' + cond_expr_expanded.__class__.__name__)
    try:
        if cond_expr_expanded in cached_conditions_asts:
            ast_code = cached_conditions_asts[cond_expr_expanded]
        else:
            ast_code = compile(cond_expr_expanded, '<string>', 'eval')
            cached_conditions_asts[cond_expr_expanded] = ast_code
        env = {'__builtins__': {}, 'v': StrictVersion}
        if eval(ast_code, env, variables):
            return true_dict
        return false_dict
    except SyntaxError as e:
        syntax_error = SyntaxError("%s while evaluating condition '%s' in %s at character %d." % (str(e.args[0]), e.text, build_file, e.offset), e.filename, e.lineno, e.offset, e.text)
        raise syntax_error
    except NameError as e:
        gyp.common.ExceptionAppend(e, f"while evaluating condition '{cond_expr_expanded}' in {build_file}")
        raise GypError(e)