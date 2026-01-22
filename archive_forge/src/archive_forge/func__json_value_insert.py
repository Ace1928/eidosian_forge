import datetime
import decimal
import json
import re
import uuid
from .. import config
from .. import engines
from .. import fixtures
from .. import mock
from ..assertions import eq_
from ..assertions import is_
from ..assertions import ne_
from ..config import requirements
from ..schema import Column
from ..schema import Table
from ... import and_
from ... import ARRAY
from ... import BigInteger
from ... import bindparam
from ... import Boolean
from ... import case
from ... import cast
from ... import Date
from ... import DateTime
from ... import Float
from ... import Integer
from ... import Interval
from ... import JSON
from ... import literal
from ... import literal_column
from ... import MetaData
from ... import null
from ... import Numeric
from ... import select
from ... import String
from ... import testing
from ... import Text
from ... import Time
from ... import TIMESTAMP
from ... import type_coerce
from ... import TypeDecorator
from ... import Unicode
from ... import UnicodeText
from ... import UUID
from ... import Uuid
from ...orm import declarative_base
from ...orm import Session
from ...sql import sqltypes
from ...sql.sqltypes import LargeBinary
from ...sql.sqltypes import PickleType
def _json_value_insert(self, connection, datatype, value, data_element):
    data_table = self.tables.data_table
    if datatype == '_decimal':

        class DecimalEncoder(json.JSONEncoder):

            def default(self, o):
                if isinstance(o, decimal.Decimal):
                    return str(o)
                return super().default(o)
        json_data = json.dumps(data_element, cls=DecimalEncoder)
        json_data = re.sub('"(%s)"' % str(value), str(value), json_data)
        datatype = 'numeric'
        connection.execute(data_table.insert().values(name='row1', data=bindparam(None, json_data, literal_execute=True), nulldata=bindparam(None, json_data, literal_execute=True)))
    else:
        connection.execute(data_table.insert(), {'name': 'row1', 'data': data_element, 'nulldata': data_element})
    p_s = None
    if datatype:
        if datatype == 'numeric':
            a, b = str(value).split('.')
            s = len(b)
            p = len(a) + s
            if isinstance(value, decimal.Decimal):
                compare_value = value
            else:
                compare_value = decimal.Decimal(str(value))
            p_s = (p, s)
        else:
            compare_value = value
    else:
        compare_value = value
    return (datatype, compare_value, p_s)