import gzip
import io
import struct
def _lz4_compress(payload, **kwargs):
    kwargs.pop('block_linked', None)
    return lz4.compress(payload, block_linked=False, **kwargs)