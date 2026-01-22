import inspect
import operator
import typing as t
from collections import deque
from markupsafe import Markup
from .utils import _PassArg
class AssignBlock(Stmt):
    """Assigns a block to a target."""
    fields = ('target', 'filter', 'body')
    target: 'Expr'
    filter: t.Optional['Filter']
    body: t.List[Node]