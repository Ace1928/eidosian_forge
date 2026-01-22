from breezy import tests
from breezy.bzr import groupcompress
class TestCaseWithInventory(tests.TestCaseWithMemoryTransport):
    _inventory_class = None
    _inv_to_test_inv = None

    def make_test_inventory(self):
        """Return an instance of the Inventory class under test."""
        return self._inventory_class()

    def inv_to_test_inv(self, inv):
        """Convert a regular Inventory object into an inventory under test."""
        return self._inv_to_test_inv(self, inv)