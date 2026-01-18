from io import BytesIO
from ... import tests
from .. import pack
def test_validate_undecodeable_name(self):
    """Names that aren't valid UTF-8 cause validate to fail."""
    reader = self.get_reader_for(b'0\n\xcc\n\n')
    self.assertRaises(pack.InvalidRecordError, reader.validate)