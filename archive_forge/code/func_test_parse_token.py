import base64
import os
import tempfile
from oslo_config import cfg
import webob
from oslo_middleware import basic_auth as auth
from oslotest import base as test_base
def test_parse_token(self):
    token = base64.b64encode(b'myName:myPassword')
    self.assertEqual(('myName', b'myPassword'), auth.parse_token(token))
    token = str(token, encoding='utf-8')
    self.assertEqual(('myName', b'myPassword'), auth.parse_token(token))
    e = self.assertRaises(webob.exc.HTTPBadRequest, auth.parse_token, token[:-1])
    self.assertEqual('Could not decode authorization token', str(e))
    token = str(base64.b64encode(b'myNamemyPassword'), encoding='utf-8')
    e = self.assertRaises(webob.exc.HTTPBadRequest, auth.parse_token, token[:-1])
    self.assertEqual('Could not decode authorization token', str(e))