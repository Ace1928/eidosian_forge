from winappdbg.textio import HexInput
from winappdbg.util import StaticClass, MemoryAddresses
from winappdbg import win32
import warnings
class TextPattern(BytePattern):
    """
    Text pattern.

    @type isUnicode: bool
    @ivar isUnicode: C{True} if the text to search for is a compat.unicode string,
        C{False} otherwise.

    @type encoding: str
    @ivar encoding: Encoding for the text parameter.
        Only used when the text to search for is a Unicode string.
        Don't change unless you know what you're doing!

    @type caseSensitive: bool
    @ivar caseSensitive: C{True} of the search is case sensitive,
        C{False} otherwise.
    """

    def __init__(self, text, encoding='utf-16le', caseSensitive=False):
        """
        @type  text: str or compat.unicode
        @param text: Text to search for.

        @type  encoding: str
        @param encoding: (Optional) Encoding for the text parameter.
            Only used when the text to search for is a Unicode string.
            Don't change unless you know what you're doing!

        @type  caseSensitive: bool
        @param caseSensitive: C{True} of the search is case sensitive,
            C{False} otherwise.
        """
        self.isUnicode = isinstance(text, compat.unicode)
        self.encoding = encoding
        self.caseSensitive = caseSensitive
        if not self.caseSensitive:
            pattern = text.lower()
        if self.isUnicode:
            pattern = text.encode(encoding)
        super(TextPattern, self).__init__(pattern)

    def read(self, process, address, size):
        data = super(TextPattern, self).read(address, size)
        if not self.caseSensitive:
            if self.isUnicode:
                try:
                    encoding = self.encoding
                    text = data.decode(encoding, 'replace')
                    text = text.lower()
                    new_data = text.encode(encoding, 'replace')
                    if len(data) == len(new_data):
                        data = new_data
                    else:
                        data = data.lower()
                except Exception:
                    data = data.lower()
            else:
                data = data.lower()
        return data

    def found(self, address, size, data):
        if self.isUnicode:
            try:
                data = compat.unicode(data, self.encoding)
            except Exception:
                return None
        return (address, size, data)