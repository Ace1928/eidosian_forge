from llvmlite.ir import types
from llvmlite.ir.values import (Block, Function, Value, NamedValue, Constant,
from llvmlite.ir._utils import _HasMetadata
class FilterClause(_LandingPadClause):
    kind = 'filter'

    def __init__(self, value):
        assert isinstance(value, Constant)
        assert isinstance(value.type, types.ArrayType)
        super(FilterClause, self).__init__(value)