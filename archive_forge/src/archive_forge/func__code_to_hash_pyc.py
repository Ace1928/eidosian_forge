import _imp
import _io
import sys
import _warnings
import marshal
def _code_to_hash_pyc(code, source_hash, checked=True):
    """Produce the data for a hash-based pyc."""
    data = bytearray(MAGIC_NUMBER)
    flags = 1 | checked << 1
    data.extend(_pack_uint32(flags))
    assert len(source_hash) == 8
    data.extend(source_hash)
    data.extend(marshal.dumps(code))
    return data