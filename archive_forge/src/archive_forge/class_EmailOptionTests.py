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
class EmailOptionTests(tests.TestCase):

    def test_default_email_uses_BRZ_EMAIL(self):
        conf = config.MemoryStack(b'email=jelmer@debian.org')
        self.overrideEnv('BRZ_EMAIL', 'jelmer@samba.org')
        self.overrideEnv('BZR_EMAIL', 'jelmer@jelmer.uk')
        self.overrideEnv('EMAIL', 'jelmer@apache.org')
        self.assertEqual('jelmer@samba.org', conf.get('email'))

    def test_default_email_uses_BZR_EMAIL(self):
        conf = config.MemoryStack(b'email=jelmer@debian.org')
        self.overrideEnv('BZR_EMAIL', 'jelmer@samba.org')
        self.overrideEnv('EMAIL', 'jelmer@apache.org')
        self.assertEqual('jelmer@samba.org', conf.get('email'))

    def test_default_email_uses_EMAIL(self):
        conf = config.MemoryStack(b'')
        self.overrideEnv('BRZ_EMAIL', None)
        self.overrideEnv('EMAIL', 'jelmer@apache.org')
        self.assertEqual('jelmer@apache.org', conf.get('email'))

    def test_BRZ_EMAIL_overrides(self):
        conf = config.MemoryStack(b'email=jelmer@debian.org')
        self.overrideEnv('BRZ_EMAIL', 'jelmer@apache.org')
        self.assertEqual('jelmer@apache.org', conf.get('email'))
        self.overrideEnv('BRZ_EMAIL', None)
        self.overrideEnv('EMAIL', 'jelmer@samba.org')
        self.assertEqual('jelmer@debian.org', conf.get('email'))