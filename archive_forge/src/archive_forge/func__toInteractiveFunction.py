import contextlib
import functools
import inspect
import pydoc
from .. import functions as fn
from . import Parameter
from .parameterTypes import ActionGroupParameter
@staticmethod
def _toInteractiveFunction(function):
    if isinstance(function, InteractiveFunction):
        return function
    interactive = InteractiveFunction(function)
    refOwner = function if not inspect.ismethod(function) else function.__func__
    if hasattr(refOwner, 'interactiveRefs'):
        refOwner.interactiveRefs.append(interactive)
    else:
        refOwner.interactiveRefs = [interactive]
    return interactive