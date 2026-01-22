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
def _large_dec_to_string(self, value):
    _int = value.as_tuple()[1]
    if 'E' in str(value):
        result = '%s%s%s' % (value < 0 and '-' or '', ''.join([str(s) for s in _int]), '0' * (value.adjusted() - (len(_int) - 1)))
    elif len(_int) - 1 > value.adjusted():
        result = '%s%s.%s' % (value < 0 and '-' or '', ''.join([str(s) for s in _int][0:value.adjusted() + 1]), ''.join([str(s) for s in _int][value.adjusted() + 1:]))
    else:
        result = '%s%s' % (value < 0 and '-' or '', ''.join([str(s) for s in _int][0:value.adjusted() + 1]))
    return result