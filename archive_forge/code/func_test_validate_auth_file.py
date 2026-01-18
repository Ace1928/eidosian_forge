import base64
import os
import tempfile
from oslo_config import cfg
import webob
from oslo_middleware import basic_auth as auth
from oslotest import base as test_base
def test_validate_auth_file(self):
    auth_file = self.write_auth_file('myName:$2y$05$lE3eGtyj41jZwrzS87KTqe6.JETVCWBkc32C63UP2aYrGoYOEpbJm\n\n\n')
    auth.validate_auth_file(auth_file)
    auth_file = auth_file + '.missing'
    self.assertRaises(auth.ConfigInvalid, auth.validate_auth_file, auth_file)
    auth_file = self.write_auth_file('foo:bar\nmyName:$2y$05$lE3eGtyj41jZwrzS87KTqe6.JETVCWBkc32C63UP2aYrGoYOEpbJm\n\n\n')
    self.assertRaises(webob.exc.HTTPBadRequest, auth.validate_auth_file, auth_file)