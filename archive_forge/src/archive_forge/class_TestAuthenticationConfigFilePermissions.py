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
class TestAuthenticationConfigFilePermissions(tests.TestCaseInTempDir):
    """Test warning for permissions of authentication.conf."""

    def setUp(self):
        super().setUp()
        self.path = osutils.pathjoin(self.test_dir, 'authentication.conf')
        with open(self.path, 'wb') as f:
            f.write(b'[broken]\nscheme=ftp\nuser=joe\nport=port # Error: Not an int\n')
        self.overrideAttr(bedding, 'authentication_config_path', lambda: self.path)
        osutils.chmod_if_possible(self.path, 493)

    def test_check_warning(self):
        conf = config.AuthenticationConfig()
        self.assertEqual(conf._filename, self.path)
        self.assertContainsRe(self.get_log(), 'Saved passwords may be accessible by other users.')

    def test_check_suppressed_warning(self):
        global_config = config.GlobalConfig()
        global_config.set_user_option('suppress_warnings', 'insecure_permissions')
        conf = config.AuthenticationConfig()
        self.assertEqual(conf._filename, self.path)
        self.assertNotContainsRe(self.get_log(), 'Saved passwords may be accessible by other users.')