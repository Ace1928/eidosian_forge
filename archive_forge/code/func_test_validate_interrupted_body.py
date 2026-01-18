from io import BytesIO
from ... import tests
from .. import pack
def test_validate_interrupted_body(self):
    """EOF during reading a record's body causes validate to fail."""
    reader = self.get_reader_for(b'1\n\n')
    self.assertRaises(pack.UnexpectedEndOfContainerError, reader.validate)