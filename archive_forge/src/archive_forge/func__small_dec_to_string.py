import datetime
import decimal
import re
import struct
from .base import _MSDateTime
from .base import _MSUnicode
from .base import _MSUnicodeText
from .base import BINARY
from .base import DATETIMEOFFSET
from .base import MSDialect
from .base import MSExecutionContext
from .base import VARBINARY
from .json import JSON as _MSJson
from .json import JSONIndexType as _MSJsonIndexType
from .json import JSONPathType as _MSJsonPathType
from ... import exc
from ... import types as sqltypes
from ... import util
from ...connectors.pyodbc import PyODBCConnector
from ...engine import cursor as _cursor
def _small_dec_to_string(self, value):
    return '%s0.%s%s' % (value < 0 and '-' or '', '0' * (abs(value.adjusted()) - 1), ''.join([str(nint) for nint in value.as_tuple()[1]]))