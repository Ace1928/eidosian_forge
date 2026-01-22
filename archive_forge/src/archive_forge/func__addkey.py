import ast as _ast
import io as _io
import os as _os
import collections.abc
def _addkey(self, key, pos_and_siz_pair):
    self._index[key] = pos_and_siz_pair
    with _io.open(self._dirfile, 'a', encoding='Latin-1') as f:
        self._chmod(self._dirfile)
        f.write('%r, %r\n' % (key.decode('Latin-1'), pos_and_siz_pair))