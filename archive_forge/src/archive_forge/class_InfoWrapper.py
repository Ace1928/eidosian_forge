import struct
from oslo_log import log as logging
class InfoWrapper(object):
    """A file-like object that wraps another and updates a format inspector.

    This passes chunks to the format inspector while reading. If the inspector
    fails, it logs the error and stops calling it, but continues proxying data
    from the source to its user.
    """

    def __init__(self, source, fmt):
        self._source = source
        self._format = fmt
        self._error = False

    def __iter__(self):
        return self

    def _process_chunk(self, chunk):
        if not self._error:
            try:
                self._format.eat_chunk(chunk)
            except Exception as e:
                LOG.error('Format inspector failed, aborting: %s', e)
                self._error = True

    def __next__(self):
        try:
            chunk = next(self._source)
        except StopIteration:
            raise
        self._process_chunk(chunk)
        return chunk

    def read(self, size):
        chunk = self._source.read(size)
        self._process_chunk(chunk)
        return chunk

    def close(self):
        if hasattr(self._source, 'close'):
            self._source.close()