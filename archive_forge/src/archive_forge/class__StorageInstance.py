import contextlib
import getpass
import logging
import os
import sqlite3
import tempfile
import warnings
from io import BytesIO
from os.path import join as pjoin
import numpy
from nibabel.optpkg import optional_package
from .nifti1 import Nifti1Header
class _StorageInstance:

    def __init__(self, d):
        self.uid = d['uid']
        self.instance_number = d['instance_number']
        self.series = d['series']
        self.files = None

    def __getattribute__(self, name):
        val = object.__getattribute__(self, name)
        if name == 'files' and val is None:
            with DB.readonly_cursor() as c:
                query = 'SELECT directory, name\n                             FROM file\n                            WHERE storage_instance = ?\n                            ORDER BY directory, name'
                c.execute(query, (self.uid,))
                val = ['%s/%s' % tuple(row) for row in c]
            self.files = val
        return val

    def dicom(self):
        return pydicom.dcmread(self.files[0])