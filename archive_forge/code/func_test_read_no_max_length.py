from io import BytesIO
from ... import tests
from .. import pack
def test_read_no_max_length(self):
    """If the max_length passed to the callable returned by read is None,
        then all the bytes in the record will be read.
        """
    reader = self.get_reader_for(b'6\n\nabcdef')
    names, get_bytes = reader.read()
    self.assertEqual(b'abcdef', get_bytes(None))