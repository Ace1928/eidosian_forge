import zlib
def _record_z_len(self, count):
    self._compressed_size_added += count
    self._unflushed_size_added = 0
    self._estimated_compression = float(self._uncompressed_size_added) / self._compressed_size_added