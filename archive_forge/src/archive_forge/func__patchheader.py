from collections import namedtuple
import warnings
def _patchheader(self):
    if self._form_length_pos is None:
        raise OSError('cannot seek')
    self._file.seek(self._form_length_pos)
    _write_u32(self._file, self._datawritten)
    self._datalength = self._datawritten
    self._file.seek(0, 2)