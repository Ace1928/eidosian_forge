import binascii
import importlib.util
import io
import itertools
import os
import posixpath
import shutil
import stat
import struct
import sys
import threading
import time
import contextlib
import pathlib
def _check_compression(compression):
    if compression == ZIP_STORED:
        pass
    elif compression == ZIP_DEFLATED:
        if not zlib:
            raise RuntimeError('Compression requires the (missing) zlib module')
    elif compression == ZIP_BZIP2:
        if not bz2:
            raise RuntimeError('Compression requires the (missing) bz2 module')
    elif compression == ZIP_LZMA:
        if not lzma:
            raise RuntimeError('Compression requires the (missing) lzma module')
    else:
        raise NotImplementedError('That compression method is not supported')