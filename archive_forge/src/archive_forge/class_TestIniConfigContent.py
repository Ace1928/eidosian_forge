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
class TestIniConfigContent(tests.TestCaseWithTransport):
    """Simulate loading a IniBasedConfig with content of various encodings.

    All files produced by bzr are in utf8 content.

    Users may modify them manually and end up with a file that can't be
    loaded. We need to issue proper error messages in this case.
    """
    invalid_utf8_char = b'\xff'

    def test_load_utf8(self):
        """Ensure we can load an utf8-encoded file."""
        unicode_user = 'b€ar'
        unicode_content = 'user={}'.format(unicode_user)
        utf8_content = unicode_content.encode('utf8')
        with open('foo.conf', 'wb') as f:
            f.write(utf8_content)
        conf = config.IniBasedConfig(file_name='foo.conf')
        self.assertEqual(unicode_user, conf.get_user_option('user'))

    def test_load_badly_encoded_content(self):
        """Ensure we display a proper error on non-ascii, non utf-8 content."""
        with open('foo.conf', 'wb') as f:
            f.write(b'user=foo\n#%s\n' % (self.invalid_utf8_char,))
        conf = config.IniBasedConfig(file_name='foo.conf')
        self.assertRaises(config.ConfigContentError, conf._get_parser)

    def test_load_erroneous_content(self):
        """Ensure we display a proper error on content that can't be parsed."""
        with open('foo.conf', 'wb') as f:
            f.write(b'[open_section\n')
        conf = config.IniBasedConfig(file_name='foo.conf')
        self.assertRaises(config.ParseConfigError, conf._get_parser)