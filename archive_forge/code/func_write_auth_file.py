import base64
import os
import tempfile
from oslo_config import cfg
import webob
from oslo_middleware import basic_auth as auth
from oslotest import base as test_base
def write_auth_file(self, data=None):
    if not data:
        data = '\n'
    with tempfile.NamedTemporaryFile(mode='w', delete=False) as f:
        f.write(data)
        self.addCleanup(os.remove, f.name)
        return f.name