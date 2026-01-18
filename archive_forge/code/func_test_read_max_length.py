from io import BytesIO
from ... import tests
from .. import pack
def test_read_max_length(self):
    """If the max_length passed to the callable returned by read is not
        None, then no more than that many bytes will be read.
        """
    reader = self.get_reader_for(b'6\n\nabcdef')
    names, get_bytes = reader.read()
    self.assertEqual(b'abc', get_bytes(3))