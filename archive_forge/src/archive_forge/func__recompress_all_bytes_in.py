import zlib
from typing import Callable, List, Optional, Tuple
from zlib import Z_FINISH, Z_SYNC_FLUSH
def _recompress_all_bytes_in(self, extra_bytes: Optional[bytes]=None) -> Tuple[List[bytes], int, 'zlib._Compress']:
    """Recompress the current bytes_in, and optionally more.

        :param extra_bytes: Optional, if supplied we will add it with
            Z_SYNC_FLUSH
        :return: (bytes_out, bytes_out_len, alt_compressed)

            * bytes_out: is the compressed bytes returned from the compressor
            * bytes_out_len: the length of the compressed output
            * compressor: An object with everything packed in so far, and
              Z_SYNC_FLUSH called.
        """
    compressor = zlib.compressobj()
    bytes_out: List[bytes] = []
    append = bytes_out.append
    compress = compressor.compress
    for accepted_bytes in self.bytes_in:
        out = compress(accepted_bytes)
        if out:
            append(out)
    if extra_bytes:
        out = compress(extra_bytes)
        out += compressor.flush(Z_SYNC_FLUSH)
        append(out)
    bytes_out_len = sum(map(len, bytes_out))
    return (bytes_out, bytes_out_len, compressor)