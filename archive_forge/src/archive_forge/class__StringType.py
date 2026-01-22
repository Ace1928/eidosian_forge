import datetime
from ... import exc
from ... import util
from ...sql import sqltypes
class _StringType(sqltypes.String):
    """Base for MySQL string types."""

    def __init__(self, charset=None, collation=None, ascii=False, binary=False, unicode=False, national=False, **kw):
        self.charset = charset
        kw.setdefault('collation', kw.pop('collate', collation))
        self.ascii = ascii
        self.unicode = unicode
        self.binary = binary
        self.national = national
        super().__init__(**kw)

    def __repr__(self):
        return util.generic_repr(self, to_inspect=[_StringType, sqltypes.String])