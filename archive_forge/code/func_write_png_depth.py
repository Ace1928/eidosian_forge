import binascii
import struct
from typing import Optional
def write_png_depth(filename: str, depth: int) -> None:
    """Write the special tEXt chunk indicating the depth to a PNG file.

    The chunk is placed immediately before the special IEND chunk.
    """
    data = struct.pack('!i', depth)
    with open(filename, 'r+b') as f:
        f.seek(-LEN_IEND, 2)
        f.write(DEPTH_CHUNK_LEN + DEPTH_CHUNK_START + data)
        crc = binascii.crc32(DEPTH_CHUNK_START + data) & 4294967295
        f.write(struct.pack('!I', crc))
        f.write(IEND_CHUNK)