import threading
import fasteners
from fasteners import test
class DecoratorsTest(test.TestCase):

    def test_locked(self):
        obj = Locked()
        obj.i_am_locked(lambda is_locked: self.assertTrue(is_locked))
        obj.i_am_not_locked(lambda is_locked: self.assertFalse(is_locked))

    def test_many_locked(self):
        obj = ManyLocks(10)
        obj.i_am_locked(lambda gotten: self.assertTrue(all(gotten)))
        obj.i_am_not_locked(lambda gotten: self.assertEqual(0, sum(gotten)))

    def test_read_write_locked(self):
        reader = fasteners.ReaderWriterLock.READER
        writer = fasteners.ReaderWriterLock.WRITER
        obj = RWLocked()
        obj.i_am_write_locked(lambda owner: self.assertEqual(owner, writer))
        obj.i_am_read_locked(lambda owner: self.assertEqual(owner, reader))
        obj.i_am_not_locked(lambda owner: self.assertIsNone(owner))