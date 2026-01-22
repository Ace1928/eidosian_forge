import breezy.errors as errors
import breezy.transactions as transactions
from breezy.tests import TestCase
class TestPassThroughTransaction(TestCase):

    def test_construct(self):
        transactions.PassThroughTransaction()

    def test_register_clean(self):
        transaction = transactions.PassThroughTransaction()
        transaction.register_clean('anobject')

    def test_register_dirty(self):
        transaction = transactions.PassThroughTransaction()
        transaction.register_dirty('anobject')

    def test_map(self):
        transaction = transactions.PassThroughTransaction()
        self.assertNotEqual(None, getattr(transaction, 'map', None))

    def test_add_and_get(self):
        transaction = transactions.PassThroughTransaction()
        weave = 'a weave'
        transaction.map.add_weave('id', weave)
        self.assertEqual(None, transaction.map.find_weave('id'))

    def test_finish_returns(self):
        transaction = transactions.PassThroughTransaction()
        transaction.finish()

    def test_finish_tells_versioned_file_finished(self):
        weave = DummyWeave('a weave')
        transaction = transactions.PassThroughTransaction()
        transaction.register_dirty(weave)
        transaction.finish()
        self.assertTrue(weave.finished)

    def test_cache_is_ignored(self):
        transaction = transactions.PassThroughTransaction()
        transaction.set_cache_size(100)
        weave = 'a weave'
        transaction.map.add_weave('id', weave)
        self.assertEqual(None, transaction.map.find_weave('id'))

    def test_writable(self):
        transaction = transactions.PassThroughTransaction()
        self.assertTrue(transaction.writeable())