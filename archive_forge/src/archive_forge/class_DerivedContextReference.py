import inspect
import operator
import typing as t
from collections import deque
from markupsafe import Markup
from .utils import _PassArg
class DerivedContextReference(Expr):
    """Return the current template context including locals. Behaves
    exactly like :class:`ContextReference`, but includes local
    variables, such as from a ``for`` loop.

    .. versionadded:: 2.11
    """