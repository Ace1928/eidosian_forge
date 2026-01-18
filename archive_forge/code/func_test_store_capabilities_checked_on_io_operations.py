from glance_store import capabilities as caps
from glance_store.tests import base
def test_store_capabilities_checked_on_io_operations(self):
    self.assertEqual('op_checker', self.store.add.__name__)
    self.assertEqual('op_checker', self.store.get.__name__)
    self.assertEqual('op_checker', self.store.delete.__name__)