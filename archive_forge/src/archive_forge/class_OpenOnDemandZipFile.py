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
class OpenOnDemandZipFile(zipfile.ZipFile):
    """
    A subclass of ``zipfile.ZipFile`` that closes its file pointer
    whenever it is not using it; and re-opens it when it needs to read
    data from the zipfile.  This is useful for reducing the number of
    open file handles when many zip files are being accessed at once.
    ``OpenOnDemandZipFile`` must be constructed from a filename, not a
    file-like object (to allow re-opening).  ``OpenOnDemandZipFile`` is
    read-only (i.e. ``write()`` and ``writestr()`` are disabled.
    """

    @py3_data
    def __init__(self, filename):
        if not isinstance(filename, str):
            raise TypeError('ReopenableZipFile filename must be a string')
        zipfile.ZipFile.__init__(self, filename)
        assert self.filename == filename
        self.close()
        self._fileRefCnt = 0

    def read(self, name):
        assert self.fp is None
        self.fp = open(self.filename, 'rb')
        value = zipfile.ZipFile.read(self, name)
        self._fileRefCnt += 1
        self.close()
        return value

    def write(self, *args, **kwargs):
        """:raise NotImplementedError: OpenOnDemandZipfile is read-only"""
        raise NotImplementedError('OpenOnDemandZipfile is read-only')

    def writestr(self, *args, **kwargs):
        """:raise NotImplementedError: OpenOnDemandZipfile is read-only"""
        raise NotImplementedError('OpenOnDemandZipfile is read-only')

    def __repr__(self):
        return repr('OpenOnDemandZipFile(%r)' % self.filename)