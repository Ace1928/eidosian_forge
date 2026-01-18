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
def test_set_extended_int(self):
    r = self.app.post('/argtypes/setextendedint', '{"value": 3}', headers={'Content-Type': 'application/json'})
    self.assertEqual(r.status_int, 200)
    self.assertEqual(r.json, 3)