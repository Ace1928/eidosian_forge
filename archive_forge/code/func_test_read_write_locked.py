import threading
import fasteners
from fasteners import test
def test_read_write_locked(self):
    reader = fasteners.ReaderWriterLock.READER
    writer = fasteners.ReaderWriterLock.WRITER
    obj = RWLocked()
    obj.i_am_write_locked(lambda owner: self.assertEqual(owner, writer))
    obj.i_am_read_locked(lambda owner: self.assertEqual(owner, reader))
    obj.i_am_not_locked(lambda owner: self.assertIsNone(owner))