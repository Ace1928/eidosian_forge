import datetime
from ... import exc
from ... import util
from ...sql import sqltypes
class MEDIUMBLOB(sqltypes._Binary):
    """MySQL MEDIUMBLOB type, for binary data up to 2^24 bytes."""
    __visit_name__ = 'MEDIUMBLOB'