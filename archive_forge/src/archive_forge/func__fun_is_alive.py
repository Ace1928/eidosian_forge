import inspect
import sys
from collections import deque
from weakref import WeakMethod, ref
from .abstract import Thenable
from .utils import reraise
def _fun_is_alive(self, fun):
    return fun() if self.weak else self.fun