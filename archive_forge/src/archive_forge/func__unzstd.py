import zlib
from io import BytesIO
def _unzstd(data: bytes, *, max_size: int=0) -> bytes:
    decompressor = zstandard.ZstdDecompressor()
    stream_reader = decompressor.stream_reader(BytesIO(data))
    output_stream = BytesIO()
    output_chunk = b'.'
    decompressed_size = 0
    while output_chunk:
        output_chunk = stream_reader.read(_CHUNK_SIZE)
        decompressed_size += len(output_chunk)
        if max_size and decompressed_size > max_size:
            raise _DecompressionMaxSizeExceeded(f'The number of bytes decompressed so far ({decompressed_size} B) exceed the specified maximum ({max_size} B).')
        output_stream.write(output_chunk)
    output_stream.seek(0)
    return output_stream.read()