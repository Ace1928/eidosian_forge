import breezy.errors as errors
import breezy.transactions as transactions
from breezy.tests import TestCase
def test_register_dirty(self):
    self.transaction.register_dirty('anobject')