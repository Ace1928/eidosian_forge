from io import BytesIO
from ... import tests
from .. import pack
def test_validate_duplicate_name(self):
    """validate raises DuplicateRecordNameError if the same name occurs
        multiple times in the container.
        """
    reader = self.get_reader_for(b'Bazaar pack format 1 (introduced in 0.18)\nB0\nname\n\nB0\nname\n\nE')
    self.assertRaises(pack.DuplicateRecordNameError, reader.validate)