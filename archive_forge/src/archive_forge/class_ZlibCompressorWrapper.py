import io
import zlib
from joblib.backports import LooseVersion
class ZlibCompressorWrapper(CompressorWrapper):

    def __init__(self):
        CompressorWrapper.__init__(self, obj=BinaryZlibFile, prefix=_ZLIB_PREFIX, extension='.z')