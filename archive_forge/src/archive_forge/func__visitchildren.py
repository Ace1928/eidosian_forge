from __future__ import absolute_import, print_function
import sys
import inspect
from . import TypeSlots
from . import Builtin
from . import Nodes
from . import ExprNodes
from . import Errors
from . import DebugFlags
from . import Future
import cython
@cython.final
@cython.locals(idx=cython.Py_ssize_t)
def _visitchildren(self, parent, attrs, exclude):
    """
        Visits the children of the given parent. If parent is None, returns
        immediately (returning None).

        The return value is a dictionary giving the results for each
        child (mapping the attribute name to either the return value
        or a list of return values (in the case of multiple children
        in an attribute)).
        """
    if parent is None:
        return None
    result = {}
    for attr in parent.child_attrs:
        if attrs is not None and attr not in attrs:
            continue
        if exclude is not None and attr in exclude:
            continue
        child = getattr(parent, attr)
        if child is not None:
            if type(child) is list:
                childretval = [self._visitchild(x, parent, attr, idx) for idx, x in enumerate(child)]
            else:
                childretval = self._visitchild(child, parent, attr, None)
                assert not isinstance(childretval, list), 'Cannot insert list here: %s in %r' % (attr, parent)
            result[attr] = childretval
    return result