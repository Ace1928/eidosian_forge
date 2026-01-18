from __future__ import annotations
import csv
import hashlib
import os.path
import re
import stat
import time
from io import StringIO, TextIOWrapper
from zipfile import ZIP_DEFLATED, ZipFile, ZipInfo
from wheel.cli import WheelError
from wheel.util import log, urlsafe_b64decode, urlsafe_b64encode
def writestr(self, zinfo_or_arcname, data, compress_type=None):
    if isinstance(zinfo_or_arcname, str):
        zinfo_or_arcname = ZipInfo(zinfo_or_arcname, date_time=get_zipinfo_datetime())
        zinfo_or_arcname.compress_type = self.compression
        zinfo_or_arcname.external_attr = (436 | stat.S_IFREG) << 16
    if isinstance(data, str):
        data = data.encode('utf-8')
    ZipFile.writestr(self, zinfo_or_arcname, data, compress_type)
    fname = zinfo_or_arcname.filename if isinstance(zinfo_or_arcname, ZipInfo) else zinfo_or_arcname
    log.info(f"adding '{fname}'")
    if fname != self.record_path:
        hash_ = self._default_algorithm(data)
        self._file_hashes[fname] = (hash_.name, urlsafe_b64encode(hash_.digest()).decode('ascii'))
        self._file_sizes[fname] = len(data)