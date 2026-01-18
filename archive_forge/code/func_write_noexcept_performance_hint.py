from __future__ import absolute_import
import copy
import hashlib
import re
from functools import partial
from itertools import product
from Cython.Utils import cached_function
from .Code import UtilityCode, LazyUtilityCode, TempitaUtilityCode
from . import StringEncoding
from . import Naming
from .Errors import error, CannotSpecialize, performance_hint
def write_noexcept_performance_hint(pos, env, function_name=None, void_return=False, is_call=False, is_from_pxd=False):
    if function_name:
        function_name = "'%s'" % function_name
    if is_call:
        on_what = 'after calling %s ' % (function_name or 'function')
    elif function_name:
        on_what = 'on %s ' % function_name
    else:
        on_what = ''
    msg = 'Exception check %swill always require the GIL to be acquired.' % on_what
    the_function = function_name if function_name else 'the function'
    if is_call and (not function_name):
        the_function = the_function + ' you are calling'
    solutions = ["Declare %s as 'noexcept' if you control the definition and you're sure you don't want the function to raise exceptions." % the_function]
    if void_return:
        solutions.append("Use an 'int' return type on %s to allow an error code to be returned." % the_function)
    if is_from_pxd and (not void_return):
        solutions.append('Declare any exception value explicitly for functions in pxd files.')
    if len(solutions) == 1:
        msg = '%s %s' % (msg, solutions[0])
    else:
        solutions = ['\t%s. %s' % (i + 1, s) for i, s in enumerate(solutions)]
        msg = '%s\nPossible solutions:\n%s' % (msg, '\n'.join(solutions))
    performance_hint(pos, msg, env)