from __future__ import print_function, absolute_import, division, unicode_literals
import sys
import os
import datetime
import traceback
import platform  # NOQA
from _ast import *  # NOQA
from ast import parse  # NOQA
from setuptools import setup, Extension, Distribution  # NOQA
from setuptools.command import install_lib  # NOQA
from setuptools.command.sdist import sdist as _sdist  # NOQA
class InMemoryZipFile(object):

    def __init__(self, file_name=None):
        try:
            from cStringIO import StringIO
        except ImportError:
            from io import BytesIO as StringIO
        import zipfile
        self.zip_file = zipfile
        self._file_name = file_name
        self.in_memory_data = StringIO()
        self.in_memory_zip = self.zip_file.ZipFile(self.in_memory_data, 'w', self.zip_file.ZIP_DEFLATED, False)
        self.in_memory_zip.debug = 3

    def append(self, filename_in_zip, file_contents):
        """Appends a file with name filename_in_zip and contents of
        file_contents to the in-memory zip."""
        self.in_memory_zip.writestr(filename_in_zip, file_contents)
        return self

    def write_to_file(self, filename):
        """Writes the in-memory zip to a file."""
        for zfile in self.in_memory_zip.filelist:
            zfile.create_system = 0
        self.in_memory_zip.close()
        with open(filename, 'wb') as f:
            f.write(self.in_memory_data.getvalue())

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        if self._file_name is None:
            return
        self.write_to_file(self._file_name)

    def delete_from_zip_file(self, pattern=None, file_names=None):
        """
        zip_file can be a string or a zipfile.ZipFile object, the latter will be closed
        any name in file_names is deleted, all file_names provided have to be in the ZIP
        archive or else an IOError is raised
        """
        if pattern and isinstance(pattern, string_type):
            import re
            pattern = re.compile(pattern)
        if file_names:
            if not isinstance(file_names, list):
                file_names = [file_names]
        else:
            file_names = []
        with self.zip_file.ZipFile(self._file_name) as zf:
            for l in zf.infolist():
                if l.filename in file_names:
                    file_names.remove(l.filename)
                    continue
                if pattern and pattern.match(l.filename):
                    continue
                self.append(l.filename, zf.read(l))
            if file_names:
                raise IOError('[Errno 2] No such file{}: {}'.format('' if len(file_names) == 1 else 's', ', '.join([repr(f) for f in file_names])))