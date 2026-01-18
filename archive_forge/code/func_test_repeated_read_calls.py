from io import BytesIO
from ... import tests
from .. import pack
def test_repeated_read_calls(self):
    """Repeated calls to the callable returned from BytesRecordReader.read
        will not read beyond the end of the record.
        """
    reader = self.get_reader_for(b'6\n\nabcdefB3\nnext-record\nXXX')
    names, get_bytes = reader.read()
    self.assertEqual(b'abcdef', get_bytes(None))
    self.assertEqual(b'', get_bytes(None))
    self.assertEqual(b'', get_bytes(99))