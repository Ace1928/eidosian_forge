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
def test_parse_unexpected_attribute(self):
    o = {'id': '1', 'name': 'test', 'other': 'unknown', 'other2': 'still unknown'}
    for ba in (True, False):
        jd = o if ba else {'o': o}
        try:
            parse(json.dumps(jd), {'o': Obj}, ba)
            raise AssertionError('Object should not parse correcty.')
        except wsme.exc.UnknownAttribute as e:
            self.assertEqual(e.attributes, set(['other', 'other2']))