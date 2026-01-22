import io
import zlib
from joblib.backports import LooseVersion
class XZCompressorWrapper(LZMACompressorWrapper):
    prefix = _XZ_PREFIX
    extension = '.xz'
    _lzma_format_name = 'FORMAT_XZ'