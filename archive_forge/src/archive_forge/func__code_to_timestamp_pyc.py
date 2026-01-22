import _imp
import _io
import sys
import _warnings
import marshal
def _code_to_timestamp_pyc(code, mtime=0, source_size=0):
    """Produce the data for a timestamp-based pyc."""
    data = bytearray(MAGIC_NUMBER)
    data.extend(_pack_uint32(0))
    data.extend(_pack_uint32(mtime))
    data.extend(_pack_uint32(source_size))
    data.extend(marshal.dumps(code))
    return data