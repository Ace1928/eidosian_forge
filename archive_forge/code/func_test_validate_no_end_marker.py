from io import BytesIO
from ... import tests
from .. import pack
def test_validate_no_end_marker(self):
    """validate raises UnexpectedEndOfContainerError if there's no end of
        container marker, even if the container up to this point has been
        valid.
        """
    reader = self.get_reader_for(b'Bazaar pack format 1 (introduced in 0.18)\n')
    self.assertRaises(pack.UnexpectedEndOfContainerError, reader.validate)