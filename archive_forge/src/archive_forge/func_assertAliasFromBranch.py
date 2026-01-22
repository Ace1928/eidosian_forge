from .. import transport, urlutils
from ..directory_service import (AliasDirectory, DirectoryServiceRegistry,
from . import TestCase, TestCaseWithTransport
def assertAliasFromBranch(self, setter, value, alias):
    setter(value)
    self.assertEqual(value, directories.dereference(alias))