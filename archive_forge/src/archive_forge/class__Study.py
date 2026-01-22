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
class _Study:

    def __init__(self, d):
        self.uid = d['uid']
        self.date = d['date']
        self.time = d['time']
        self.comments = d['comments']
        self.patient_name = d['patient_name']
        self.patient_id = d['patient_id']
        self.patient_birth_date = d['patient_birth_date']
        self.patient_sex = d['patient_sex']
        self.series = None

    def __getattribute__(self, name):
        val = object.__getattribute__(self, name)
        if name == 'series' and val is None:
            val = []
            with DB.readonly_cursor() as c:
                c.execute('SELECT * FROM series WHERE study = ?', (self.uid,))
                cols = [el[0] for el in c.description]
                for row in c:
                    d = dict(zip(cols, row))
                    val.append(_Series(d))
            self.series = val
        return val

    def patient_name_or_uid(self):
        if self.patient_name == '':
            return self.uid
        return self.patient_name