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
def test_ssh_password_emits_warning(self):
    conf = config.AuthenticationConfig(_file=BytesIO(b'\n[ssh with password]\nscheme=ssh\nhost=bar.org\nuser=jim\npassword=jimpass\n'))
    entered_password = 'typed-by-hand'
    ui.ui_factory = tests.TestUIFactory(stdin=entered_password + '\n')
    self.assertEqual(entered_password, conf.get_password('ssh', 'bar.org', user='jim'))
    self.assertContainsRe(self.get_log(), 'password ignored in section \\[ssh with password\\]')