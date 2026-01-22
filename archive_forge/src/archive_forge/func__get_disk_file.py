from __future__ import annotations
import logging
import os
import shutil
import sys
import tempfile
from email.message import Message
from enum import IntEnum
from io import BytesIO
from numbers import Number
from typing import TYPE_CHECKING
from .decoders import Base64Decoder, QuotedPrintableDecoder
from .exceptions import FileError, FormParserError, MultipartParseError, QuerystringParseError
def _get_disk_file(self):
    """This function is responsible for getting a file object on-disk for us."""
    self.logger.info('Opening a file on disk')
    file_dir = self._config.get('UPLOAD_DIR')
    keep_filename = self._config.get('UPLOAD_KEEP_FILENAME', False)
    keep_extensions = self._config.get('UPLOAD_KEEP_EXTENSIONS', False)
    delete_tmp = self._config.get('UPLOAD_DELETE_TMP', True)
    if file_dir is not None and keep_filename:
        self.logger.info('Saving with filename in: %r', file_dir)
        fname = self._file_base
        if keep_extensions:
            fname = fname + self._ext
        path = os.path.join(file_dir, fname)
        try:
            self.logger.info('Opening file: %r', path)
            tmp_file = open(path, 'w+b')
        except OSError:
            tmp_file = None
            self.logger.exception('Error opening temporary file')
            raise FileError('Error opening temporary file: %r' % path)
    else:
        options = {}
        if keep_extensions:
            ext = self._ext
            if isinstance(ext, bytes):
                ext = ext.decode(sys.getfilesystemencoding())
            options['suffix'] = ext
        if file_dir is not None:
            d = file_dir
            if isinstance(d, bytes):
                d = d.decode(sys.getfilesystemencoding())
            options['dir'] = d
        options['delete'] = delete_tmp
        self.logger.info('Creating a temporary file with options: %r', options)
        try:
            tmp_file = tempfile.NamedTemporaryFile(**options)
        except OSError:
            self.logger.exception('Error creating named temporary file')
            raise FileError('Error creating named temporary file')
        fname = tmp_file.name
        if isinstance(fname, str):
            fname = fname.encode(sys.getfilesystemencoding())
    self._actual_file_name = fname
    return tmp_file