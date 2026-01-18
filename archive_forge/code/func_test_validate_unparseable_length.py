from io import BytesIO
from ... import tests
from .. import pack
def test_validate_unparseable_length(self):
    """An unparseable record length causes validate to fail."""
    reader = self.get_reader_for(b'\n\n')
    self.assertRaises(pack.InvalidRecordError, reader.validate)