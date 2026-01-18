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
def test_valid_str_to_builtin_fromjson(self):
    types = (int, bool, float)
    value = '2'
    for t in types:
        for ba in (True, False):
            jd = '%s' if ba else '{"a": %s}'
            i = parse(jd % value, {'a': t}, ba)
            self.assertEqual(i, {'a': t(value)}, "Parsed value does not correspond for %s: %s != {'a': %s}" % (t, repr(i), repr(t(value))))
            self.assertIsInstance(i['a'], t)