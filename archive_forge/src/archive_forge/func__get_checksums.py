from __future__ import absolute_import, division, print_function
import abc
import bz2
import glob
import gzip
import io
import os
import re
import shutil
import tarfile
import zipfile
from fnmatch import fnmatch
from sys import version_info
from traceback import format_exc
from zlib import crc32
from ansible.module_utils.basic import AnsibleModule, missing_required_lib
from ansible.module_utils.common.text.converters import to_bytes, to_native
from ansible.module_utils import six
def _get_checksums(self, path):
    if HAS_LZMA:
        LZMAError = lzma.LZMAError
    else:
        LZMAError = tarfile.ReadError
    try:
        if self.format == 'xz':
            with lzma.open(_to_native_ascii(path), 'r') as f:
                archive = tarfile.open(fileobj=f)
                checksums = set(((info.name, info.chksum) for info in archive.getmembers()))
                archive.close()
        else:
            archive = tarfile.open(_to_native_ascii(path), 'r|' + self.format)
            checksums = set(((info.name, info.chksum) for info in archive.getmembers()))
            archive.close()
    except (LZMAError, tarfile.ReadError, tarfile.CompressionError):
        try:
            f = self._open_compressed_file(_to_native_ascii(path), 'r')
            checksum = 0
            while True:
                chunk = f.read(16 * 1024 * 1024)
                if not chunk:
                    break
                checksum = crc32(chunk, checksum)
            checksums = set([(b'', checksum)])
            f.close()
        except Exception:
            checksums = set()
    return checksums