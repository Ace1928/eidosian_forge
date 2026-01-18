from collections import namedtuple
import warnings
def writeframesraw(self, data):
    if not isinstance(data, (bytes, bytearray)):
        data = memoryview(data).cast('B')
    self._ensure_header_written()
    if self._comptype == 'ULAW':
        with warnings.catch_warnings():
            warnings.simplefilter('ignore', category=DeprecationWarning)
            import audioop
        data = audioop.lin2ulaw(data, self._sampwidth)
    nframes = len(data) // self._framesize
    self._file.write(data)
    self._nframeswritten = self._nframeswritten + nframes
    self._datawritten = self._datawritten + len(data)