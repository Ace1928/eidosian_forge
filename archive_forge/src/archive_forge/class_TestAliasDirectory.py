from .. import transport, urlutils
from ..directory_service import (AliasDirectory, DirectoryServiceRegistry,
from . import TestCase, TestCaseWithTransport
class TestAliasDirectory(TestCaseWithTransport):

    def setUp(self):
        super().setUp()
        self.branch = self.make_branch('.')

    def assertAliasFromBranch(self, setter, value, alias):
        setter(value)
        self.assertEqual(value, directories.dereference(alias))

    def test_lookup_parent(self):
        self.assertAliasFromBranch(self.branch.set_parent, 'http://a', ':parent')

    def test_lookup_submit(self):
        self.assertAliasFromBranch(self.branch.set_submit_branch, 'http://b', ':submit')

    def test_lookup_public(self):
        self.assertAliasFromBranch(self.branch.set_public_branch, 'http://c', ':public')

    def test_lookup_bound(self):
        self.assertAliasFromBranch(self.branch.set_bound_location, 'http://d', ':bound')

    def test_lookup_push(self):
        self.assertAliasFromBranch(self.branch.set_push_location, 'http://e', ':push')

    def test_lookup_this(self):
        self.assertEqual(self.branch.base, directories.dereference(':this'))

    def test_extra_path(self):
        self.assertEqual(urlutils.join(self.branch.base, 'arg'), directories.dereference(':this/arg'))

    def test_lookup_badname(self):
        e = self.assertRaises(InvalidLocationAlias, directories.dereference, ':booga')
        self.assertEqual('":booga" is not a valid location alias.', str(e))

    def test_lookup_badvalue(self):
        e = self.assertRaises(UnsetLocationAlias, directories.dereference, ':parent')
        self.assertEqual('No parent location assigned.', str(e))

    def test_register_location_alias(self):
        self.addCleanup(AliasDirectory.branch_aliases.remove, 'booga')
        AliasDirectory.branch_aliases.register('booga', lambda b: 'UHH?', help='Nobody knows')
        self.assertEqual('UHH?', directories.dereference(':booga'))