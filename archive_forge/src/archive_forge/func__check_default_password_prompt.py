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
def _check_default_password_prompt(self, expected_prompt_format, scheme, host=None, port=None, realm=None, path=None):
    if host is None:
        host = 'bar.org'
    user, password = ('jim', 'precious')
    expected_prompt = expected_prompt_format % {'scheme': scheme, 'host': host, 'port': port, 'user': user, 'realm': realm}
    ui.ui_factory = tests.TestUIFactory(stdin=password + '\n')
    conf = config.AuthenticationConfig()
    self.assertEqual(password, conf.get_password(scheme, host, user, port=port, realm=realm, path=path))
    self.assertEqual(expected_prompt, ui.ui_factory.stderr.getvalue())
    self.assertEqual('', ui.ui_factory.stdout.getvalue())