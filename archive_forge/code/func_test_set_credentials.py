import os
import sys
import threading
from io import BytesIO
from textwrap import dedent
import configobj
from testtools import matchers
from .. import (bedding, branch, config, controldir, diff, errors, lock,
from .. import registry as _mod_registry
from .. import tests, trace
from .. import transport as _mod_transport
from .. import ui, urlutils
from ..bzr import remote
from ..transport import remote as transport_remote
from . import features, scenarios, test_server
def test_set_credentials(self):
    conf = config.AuthenticationConfig()
    conf.set_credentials('name', 'host', 'user', 'scheme', 'password', 99, path='/foo', verify_certificates=False, realm='realm')
    credentials = conf.get_credentials(host='host', scheme='scheme', port=99, path='/foo', realm='realm')
    CREDENTIALS = {'name': 'name', 'user': 'user', 'password': 'password', 'verify_certificates': False, 'scheme': 'scheme', 'host': 'host', 'port': 99, 'path': '/foo', 'realm': 'realm'}
    self.assertEqual(CREDENTIALS, credentials)
    credentials_from_disk = config.AuthenticationConfig().get_credentials(host='host', scheme='scheme', port=99, path='/foo', realm='realm')
    self.assertEqual(CREDENTIALS, credentials_from_disk)