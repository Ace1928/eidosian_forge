from .. import transport, urlutils
from ..directory_service import (AliasDirectory, DirectoryServiceRegistry,
from . import TestCase, TestCaseWithTransport
class TestColocatedDirectory(TestCaseWithTransport):

    def test_lookup_non_default(self):
        default = self.make_branch('.')
        non_default = default.controldir.create_branch(name='nondefault')
        self.assertEqual(non_default.base, directories.dereference('co:nondefault'))

    def test_lookup_default(self):
        default = self.make_branch('.')
        non_default = default.controldir.create_branch(name='nondefault')
        self.assertEqual(urlutils.join_segment_parameters(default.controldir.user_url, {'branch': ''}), directories.dereference('co:'))

    def test_no_such_branch(self):
        default = self.make_branch('.')
        self.assertEqual(urlutils.join_segment_parameters(default.controldir.user_url, {'branch': 'foo'}), directories.dereference('co:foo'))