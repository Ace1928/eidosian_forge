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
def _open_to_write(self, zinfo, force_zip64=False):
    if force_zip64 and (not self._allowZip64):
        raise ValueError('force_zip64 is True, but allowZip64 was False when opening the ZIP file.')
    if self._writing:
        raise ValueError("Can't write to the ZIP file while there is another write handle open on it. Close the first handle before opening another.")
    zinfo.compress_size = 0
    zinfo.CRC = 0
    zinfo.flag_bits = 0
    if zinfo.compress_type == ZIP_LZMA:
        zinfo.flag_bits |= _MASK_COMPRESS_OPTION_1
    if not self._seekable:
        zinfo.flag_bits |= _MASK_USE_DATA_DESCRIPTOR
    if not zinfo.external_attr:
        zinfo.external_attr = 384 << 16
    zip64 = force_zip64 or zinfo.file_size * 1.05 > ZIP64_LIMIT
    if not self._allowZip64 and zip64:
        raise LargeZipFile('Filesize would require ZIP64 extensions')
    if self._seekable:
        self.fp.seek(self.start_dir)
    zinfo.header_offset = self.fp.tell()
    self._writecheck(zinfo)
    self._didModify = True
    self.fp.write(zinfo.FileHeader(zip64))
    self._writing = True
    return _ZipWriteFile(self, zinfo, zip64)