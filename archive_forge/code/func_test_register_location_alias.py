from .. import transport, urlutils
from ..directory_service import (AliasDirectory, DirectoryServiceRegistry,
from . import TestCase, TestCaseWithTransport
def test_register_location_alias(self):
    self.addCleanup(AliasDirectory.branch_aliases.remove, 'booga')
    AliasDirectory.branch_aliases.register('booga', lambda b: 'UHH?', help='Nobody knows')
    self.assertEqual('UHH?', directories.dereference(':booga'))