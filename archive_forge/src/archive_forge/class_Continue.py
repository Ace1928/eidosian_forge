import inspect
import operator
import typing as t
from collections import deque
from markupsafe import Markup
from .utils import _PassArg
class Continue(Stmt):
    """Continue a loop."""