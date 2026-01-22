import enum
from pyomo.common.dependencies import attempt_import
from pyomo.common.numeric_types import native_types
from pyomo.core.pyomoobject import PyomoObject
from pyomo.core.expr.expr_common import OperatorAssociativity
class ExpressionArgs_Mixin(object):
    __slots__ = ('_args_',)

    def __init__(self, args):
        self._args_ = args

    def nargs(self):
        return len(self._args_)

    @property
    def args(self):
        """
        Return the child nodes

        Returns
        -------
        list or tuple:
            Sequence containing only the child nodes of this node.  The
            return type depends on the node storage model.  Users are
            not permitted to change the returned data (even for the case
            of data returned as a list), as that breaks the promise of
            tree immutability.
        """
        return self._args_