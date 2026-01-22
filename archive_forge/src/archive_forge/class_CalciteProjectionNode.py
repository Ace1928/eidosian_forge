import abc
from .dataframe.utils import ColNameCodec
from .db_worker import DbTable
from .expr import BaseExpr
class CalciteProjectionNode(CalciteBaseNode):
    """
    A node to represent a projection operation.

    Parameters
    ----------
    fields : list of str
        Output column names.
    exprs : list of BaseExpr
        Output column expressions.

    Attributes
    ----------
    fields : list of str
        A list of output columns.
    exprs : list of BaseExpr
        A list of expressions describing how output columns are computed.
        Order of expression follows `fields` order.
    """

    def __init__(self, fields, exprs):
        super(CalciteProjectionNode, self).__init__('LogicalProject')
        self.fields = [ColNameCodec.encode(field) for field in fields]
        self.exprs = exprs