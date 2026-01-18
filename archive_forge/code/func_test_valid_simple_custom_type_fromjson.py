import base64
import datetime
import decimal
import json
from urllib.parse import urlencode
from wsme.exc import ClientSideError, InvalidInput
from wsme.types import isarray, isdict, isusertype, register_type
from wsme.types import UserType, ArrayType, DictType
from wsme.rest import expose, validate
from wsme.rest.json import fromjson, tojson, parse
import wsme.tests.protocol
from wsme.utils import parse_isodatetime, parse_isotime, parse_isodate
def test_valid_simple_custom_type_fromjson(self):
    value = 2
    for ba in (True, False):
        jd = '"%d"' if ba else '{"a": "%d"}'
        i = parse(jd % value, {'a': CustomInt()}, ba)
        self.assertEqual(i, {'a': 2})
        self.assertIsInstance(i['a'], int)