from io import BytesIO
from ... import tests
from .. import pack
def test_unknown_record_type(self):
    """Test the formatting of UnknownRecordTypeError."""
    e = pack.UnknownRecordTypeError('X')
    self.assertEqual("Unknown record type: 'X'", str(e))