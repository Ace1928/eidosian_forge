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
class MailClientOptionTests(tests.TestCase):

    def test_default(self):
        conf = config.MemoryStack(b'')
        client = conf.get('mail_client')
        self.assertIs(client, mail_client.DefaultMail)

    def test_evolution(self):
        conf = config.MemoryStack(b'mail_client=evolution')
        client = conf.get('mail_client')
        self.assertIs(client, mail_client.Evolution)

    def test_kmail(self):
        conf = config.MemoryStack(b'mail_client=kmail')
        client = conf.get('mail_client')
        self.assertIs(client, mail_client.KMail)

    def test_mutt(self):
        conf = config.MemoryStack(b'mail_client=mutt')
        client = conf.get('mail_client')
        self.assertIs(client, mail_client.Mutt)

    def test_thunderbird(self):
        conf = config.MemoryStack(b'mail_client=thunderbird')
        client = conf.get('mail_client')
        self.assertIs(client, mail_client.Thunderbird)

    def test_explicit_default(self):
        conf = config.MemoryStack(b'mail_client=default')
        client = conf.get('mail_client')
        self.assertIs(client, mail_client.DefaultMail)

    def test_editor(self):
        conf = config.MemoryStack(b'mail_client=editor')
        client = conf.get('mail_client')
        self.assertIs(client, mail_client.Editor)

    def test_mapi(self):
        conf = config.MemoryStack(b'mail_client=mapi')
        client = conf.get('mail_client')
        self.assertIs(client, mail_client.MAPIClient)

    def test_xdg_email(self):
        conf = config.MemoryStack(b'mail_client=xdg-email')
        client = conf.get('mail_client')
        self.assertIs(client, mail_client.XDGEmail)

    def test_unknown(self):
        conf = config.MemoryStack(b'mail_client=firebird')
        self.assertRaises(config.ConfigOptionValueError, conf.get, 'mail_client')