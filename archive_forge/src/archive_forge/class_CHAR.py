import datetime
from ... import exc
from ... import util
from ...sql import sqltypes
class CHAR(_StringType, sqltypes.CHAR):
    """MySQL CHAR type, for fixed-length character data."""
    __visit_name__ = 'CHAR'

    def __init__(self, length=None, **kwargs):
        """Construct a CHAR.

        :param length: Maximum data length, in characters.

        :param binary: Optional, use the default binary collation for the
          national character set.  This does not affect the type of data
          stored, use a BINARY type for binary data.

        :param collation: Optional, request a particular collation.  Must be
          compatible with the national character set.

        """
        super().__init__(length=length, **kwargs)

    @classmethod
    def _adapt_string_for_cast(cls, type_):
        type_ = sqltypes.to_instance(type_)
        if isinstance(type_, sqltypes.CHAR):
            return type_
        elif isinstance(type_, _StringType):
            return CHAR(length=type_.length, charset=type_.charset, collation=type_.collation, ascii=type_.ascii, binary=type_.binary, unicode=type_.unicode, national=False)
        else:
            return CHAR(length=type_.length)