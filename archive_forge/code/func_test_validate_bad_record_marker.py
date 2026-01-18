from io import BytesIO
from ... import tests
from .. import pack
def test_validate_bad_record_marker(self):
    """validate raises UnknownRecordTypeError for unrecognised record
        types.
        """
    reader = self.get_reader_for(b'Bazaar pack format 1 (introduced in 0.18)\nX')
    self.assertRaises(pack.UnknownRecordTypeError, reader.validate)