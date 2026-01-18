import os
import socket
import sys
import time
from breezy import config, controldir, errors, tests
from breezy import transport as _mod_transport
from breezy import ui
from breezy.osutils import lexists
from breezy.tests import TestCase, TestCaseWithTransport, TestSkipped, features
from breezy.tests.http_server import HttpServer
def test_sftp_doesnt_prompt_username(self):
    ui.ui_factory = tests.TestUIFactory(stdin='joe\nfoo\n')
    t = self.get_transport_for_connection(set_config=False)
    self.assertIs(None, t._get_credentials()[0])
    self.assertEqual('', ui.ui_factory.stdout.getvalue())
    self.assertEqual(0, ui.ui_factory.stdin.tell())