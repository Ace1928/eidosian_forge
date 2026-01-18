import inspect
import sys
from .__wrapt__ import FunctionWrapper
def wrap_function_wrapper(module, name, wrapper):
    return wrap_object(module, name, FunctionWrapper, (wrapper,))