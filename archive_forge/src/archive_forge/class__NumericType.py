import datetime
from ... import exc
from ... import util
from ...sql import sqltypes
class _NumericType:
    """Base for MySQL numeric types.

    This is the base both for NUMERIC as well as INTEGER, hence
    it's a mixin.

    """

    def __init__(self, unsigned=False, zerofill=False, **kw):
        self.unsigned = unsigned
        self.zerofill = zerofill
        super().__init__(**kw)

    def __repr__(self):
        return util.generic_repr(self, to_inspect=[_NumericType, sqltypes.Numeric])