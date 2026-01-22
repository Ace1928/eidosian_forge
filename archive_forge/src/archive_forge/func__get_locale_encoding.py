import os
import abc
import codecs
import errno
import stat
import sys
from _thread import allocate_lock as Lock
import io
from io import (__all__, SEEK_SET, SEEK_CUR, SEEK_END)
from _io import FileIO
def _get_locale_encoding(self):
    try:
        import locale
    except ImportError:
        return 'utf-8'
    else:
        return locale.getencoding()