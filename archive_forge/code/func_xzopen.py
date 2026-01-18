from builtins import open as bltn_open
import sys
import os
import io
import shutil
import stat
import time
import struct
import copy
import re
import warnings
@classmethod
def xzopen(cls, name, mode='r', fileobj=None, preset=None, **kwargs):
    """Open lzma compressed tar archive name for reading or writing.
           Appending is not allowed.
        """
    if mode not in ('r', 'w', 'x'):
        raise ValueError("mode must be 'r', 'w' or 'x'")
    try:
        from lzma import LZMAFile, LZMAError
    except ImportError:
        raise CompressionError('lzma module is not available') from None
    fileobj = LZMAFile(fileobj or name, mode, preset=preset)
    try:
        t = cls.taropen(name, mode, fileobj, **kwargs)
    except (LZMAError, EOFError) as e:
        fileobj.close()
        if mode == 'r':
            raise ReadError('not an lzma file') from e
        raise
    except:
        fileobj.close()
        raise
    t._extfileobj = False
    return t