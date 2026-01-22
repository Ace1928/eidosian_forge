import collections
from functools import wraps
from itertools import count
from inspect import getfullargspec as getArgsSpec
import attr
from ._core import Transitioner, Automaton
from ._introspection import preserveName
def _getArgNames(spec):
    """
    Get the name of all arguments defined in a function signature.

    The name of * and ** arguments is normalized to "*args" and "**kwargs".

    :param ArgSpec spec: A function to interrogate for a signature.
    :return: The set of all argument names in `func`s signature.
    :rtype: Set[str]
    """
    return set(spec.args + spec.kwonlyargs + (('*args',) if spec.varargs else ()) + (('**kwargs',) if spec.varkw else ()) + spec.annotations)