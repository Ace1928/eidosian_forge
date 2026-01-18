from io import BytesIO
from ... import tests
from .. import pack
def test_zero_records_written_after_begin(self):
    """After begin is written, 0 records have been written."""
    self.writer.begin()
    self.assertEqual(0, self.writer.records_written)