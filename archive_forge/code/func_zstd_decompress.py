from __future__ import annotations
import zlib
from kombu.utils.encoding import ensure_bytes
def zstd_decompress(body):
    d = zstd.ZstdDecompressor()
    return d.decompress(body)