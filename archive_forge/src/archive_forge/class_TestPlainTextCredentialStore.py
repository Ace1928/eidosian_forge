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
class TestPlainTextCredentialStore(tests.TestCase):

    def test_decode_password(self):
        r = config.credential_store_registry
        plain_text = r.get_credential_store()
        decoded = plain_text.decode_password(dict(password='secret'))
        self.assertEqual('secret', decoded)