import io
import zlib
from joblib.backports import LooseVersion
class BinaryGzipFile(BinaryZlibFile):
    """A file object providing transparent gzip (de)compression.

    If filename is a str or bytes object, it gives the name
    of the file to be opened. Otherwise, it should be a file object,
    which will be used to read or write the compressed data.

    mode can be 'rb' for reading (default) or 'wb' for (over)writing

    If mode is 'wb', compresslevel can be a number between 1
    and 9 specifying the level of compression: 1 produces the least
    compression, and 9 produces the most compression. 3 is the default.
    """
    wbits = 31