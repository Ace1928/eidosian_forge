from .utils import logger, NO_VALUE
from typing import Mapping, Iterable, Callable, Union, TypeVar, Tuple, Any, List, Set, Optional, Collection, TYPE_CHECKING
class VisitError(LarkError):
    """VisitError is raised when visitors are interrupted by an exception

    It provides the following attributes for inspection:

    Parameters:
        rule: the name of the visit rule that failed
        obj: the tree-node or token that was being processed
        orig_exc: the exception that cause it to fail

    Note: These parameters are available as attributes
    """
    obj: 'Union[Tree, Token]'
    orig_exc: Exception

    def __init__(self, rule, obj, orig_exc):
        message = 'Error trying to process rule "%s":\n\n%s' % (rule, orig_exc)
        super(VisitError, self).__init__(message)
        self.rule = rule
        self.obj = obj
        self.orig_exc = orig_exc