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
def test_verify_certificates(self):
    conf = config.AuthenticationConfig(_file=BytesIO(b'\n[self-signed]\nscheme=https\nhost=bar.org\nuser=jim\npassword=jimpass\nverify_certificates=False\n[normal]\nscheme=https\nhost=foo.net\nuser=georges\npassword=bendover\n'))
    credentials = conf.get_credentials('https', 'bar.org')
    self.assertEqual(False, credentials.get('verify_certificates'))
    credentials = conf.get_credentials('https', 'foo.net')
    self.assertEqual(True, credentials.get('verify_certificates'))