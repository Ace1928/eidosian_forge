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
def test_set_custom_object(self):
    r = self.app.post('/argtypes/setcustomobject', '{"value": {"aint": 2, "name": "test"}}', headers={'Content-Type': 'application/json'})
    self.assertEqual(r.status_int, 200)
    self.assertEqual(r.json, {'aint': 2, 'name': 'test'})