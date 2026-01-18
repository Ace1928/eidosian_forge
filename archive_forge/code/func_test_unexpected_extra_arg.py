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
def test_unexpected_extra_arg(self):
    headers = {'Content-Type': 'application/json'}
    data = {'id': 1, 'name': 'test'}
    content = json.dumps({'data': data, 'other': 'unexpected'})
    res = self.app.put('/crud', content, headers=headers, expect_errors=True)
    self.assertEqual(res.status_int, 400)