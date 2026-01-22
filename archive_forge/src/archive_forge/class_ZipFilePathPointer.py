import codecs
import functools
import os
import pickle
import re
import sys
import textwrap
import zipfile
from abc import ABCMeta, abstractmethod
from gzip import WRITE as GZ_WRITE
from gzip import GzipFile
from io import BytesIO, TextIOWrapper
from urllib.request import url2pathname, urlopen
from nltk import grammar, sem
from nltk.compat import add_py3_data, py3_data
from nltk.internals import deprecated
class ZipFilePathPointer(PathPointer):
    """
    A path pointer that identifies a file contained within a zipfile,
    which can be accessed by reading that zipfile.
    """

    @py3_data
    def __init__(self, zipfile, entry=''):
        """
        Create a new path pointer pointing at the specified entry
        in the given zipfile.

        :raise IOError: If the given zipfile does not exist, or if it
        does not contain the specified entry.
        """
        if isinstance(zipfile, str):
            zipfile = OpenOnDemandZipFile(os.path.abspath(zipfile))
        if entry:
            entry = normalize_resource_name(entry, True, '/').lstrip('/')
            try:
                zipfile.getinfo(entry)
            except Exception as e:
                if entry.endswith('/') and [n for n in zipfile.namelist() if n.startswith(entry)]:
                    pass
                else:
                    raise OSError(f'Zipfile {zipfile.filename!r} does not contain {entry!r}') from e
        self._zipfile = zipfile
        self._entry = entry

    @property
    def zipfile(self):
        """
        The zipfile.ZipFile object used to access the zip file
        containing the entry identified by this path pointer.
        """
        return self._zipfile

    @property
    def entry(self):
        """
        The name of the file within zipfile that this path
        pointer points to.
        """
        return self._entry

    def open(self, encoding=None):
        data = self._zipfile.read(self._entry)
        stream = BytesIO(data)
        if self._entry.endswith('.gz'):
            stream = GzipFile(self._entry, fileobj=stream)
        elif encoding is not None:
            stream = SeekableUnicodeStreamReader(stream, encoding)
        return stream

    def file_size(self):
        return self._zipfile.getinfo(self._entry).file_size

    def join(self, fileid):
        entry = f'{self._entry}/{fileid}'
        return ZipFilePathPointer(self._zipfile, entry)

    def __repr__(self):
        return f'ZipFilePathPointer({self._zipfile.filename!r}, {self._entry!r})'

    def __str__(self):
        return os.path.normpath(os.path.join(self._zipfile.filename, self._entry))