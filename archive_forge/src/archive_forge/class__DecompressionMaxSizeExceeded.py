import zlib
from io import BytesIO
class _DecompressionMaxSizeExceeded(ValueError):
    pass