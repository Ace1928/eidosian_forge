import datetime
from ... import exc
from ... import util
from ...sql import sqltypes
class NVARCHAR(_StringType, sqltypes.NVARCHAR):
    """MySQL NVARCHAR type.

    For variable-length character data in the server's configured national
    character set.
    """
    __visit_name__ = 'NVARCHAR'

    def __init__(self, length=None, **kwargs):
        """Construct an NVARCHAR.

        :param length: Maximum data length, in characters.

        :param binary: Optional, use the default binary collation for the
          national character set.  This does not affect the type of data
          stored, use a BINARY type for binary data.

        :param collation: Optional, request a particular collation.  Must be
          compatible with the national character set.

        """
        kwargs['national'] = True
        super().__init__(length=length, **kwargs)