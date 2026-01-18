from io import BytesIO
from ... import tests
from .. import pack
def test_unknown_container_format(self):
    """Test the formatting of UnknownContainerFormatError."""
    e = pack.UnknownContainerFormatError('bad format string')
    self.assertEqual("Unrecognised container format: 'bad format string'", str(e))