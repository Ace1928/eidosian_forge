import logging
import os.path
import sys
from .exceptions import NoSuchClassError, UnsupportedPropertyError
from .icon_cache import IconCache
def _pixmap(self, prop):
    if prop.text:
        fname = prop.text.replace('\\', '\\\\')
        if self._base_dir != '' and fname[0] != ':' and (not os.path.isabs(fname)):
            fname = os.path.join(self._base_dir, fname)
        return QtGui.QPixmap(fname)
    return None