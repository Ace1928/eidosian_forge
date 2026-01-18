from __future__ import annotations
import zlib
from kombu.utils.encoding import ensure_bytes
def zstd_compress(body):
    c = zstd.ZstdCompressor()
    return c.compress(body)