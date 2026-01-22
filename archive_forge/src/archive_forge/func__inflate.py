import zlib
from io import BytesIO
def _inflate(data: bytes, *, max_size: int=0) -> bytes:
    decompressor = zlib.decompressobj()
    raw_decompressor = zlib.decompressobj(wbits=-15)
    input_stream = BytesIO(data)
    output_stream = BytesIO()
    output_chunk = b'.'
    decompressed_size = 0
    while output_chunk:
        input_chunk = input_stream.read(_CHUNK_SIZE)
        try:
            output_chunk = decompressor.decompress(input_chunk)
        except zlib.error:
            if decompressor != raw_decompressor:
                decompressor = raw_decompressor
                output_chunk = decompressor.decompress(input_chunk)
            else:
                raise
        decompressed_size += len(output_chunk)
        if max_size and decompressed_size > max_size:
            raise _DecompressionMaxSizeExceeded(f'The number of bytes decompressed so far ({decompressed_size} B) exceed the specified maximum ({max_size} B).')
        output_stream.write(output_chunk)
    output_stream.seek(0)
    return output_stream.read()