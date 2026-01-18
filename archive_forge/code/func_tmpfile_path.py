from ._base import *
@property
def tmpfile_path(self):
    if self._tmp_file and self._method == 'tmp':
        return self._tmp_file.name
    return None