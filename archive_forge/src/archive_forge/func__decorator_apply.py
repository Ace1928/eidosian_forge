import os
from typing import Any, Callable, TypeVar
from joblib import Memory
from decorator import FunctionMaker
def _decorator_apply(dec, func):
    return FunctionMaker.create(func, 'return decfunc(%(shortsignature)s)', dict(decfunc=dec(func)), __wrapped__=func)