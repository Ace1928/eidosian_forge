from io import BytesIO
from ... import tests
from .. import pack
def test_validate_bad_format(self):
    """validate raises an error for unrecognised format strings.

        It may raise either UnexpectedEndOfContainerError or
        UnknownContainerFormatError, depending on exactly what the string is.
        """
    inputs = [b'', b'x', b'Bazaar pack format 1 (introduced in 0.18)', b'bad\n']
    for input in inputs:
        reader = self.get_reader_for(input)
        self.assertRaises((pack.UnexpectedEndOfContainerError, pack.UnknownContainerFormatError), reader.validate)