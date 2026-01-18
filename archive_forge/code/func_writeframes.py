from collections import namedtuple
import warnings
def writeframes(self, data):
    self.writeframesraw(data)
    if self._nframeswritten != self._nframes or self._datalength != self._datawritten:
        self._patchheader()