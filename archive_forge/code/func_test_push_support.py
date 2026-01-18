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
def test_push_support(self):
    self.build_tree(['a/', 'a/foo'])
    t = controldir.ControlDir.create_standalone_workingtree('a')
    b = t.branch
    t.add('foo')
    revid1 = t.commit('foo')
    b2 = controldir.ControlDir.create_branch_and_repo(self.get_url('/b'))
    b2.pull(b)
    self.assertEqual(b2.last_revision(), revid1)
    with open('a/foo', 'w') as f:
        f.write('something new in foo\n')
    revid2 = t.commit('new')
    b2.pull(b)
    self.assertEqual(b2.last_revision(), revid2)