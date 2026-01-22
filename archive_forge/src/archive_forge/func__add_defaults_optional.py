import os
import sys
from glob import glob
from warnings import warn
from distutils.core import Command
from distutils import dir_util
from distutils import file_util
from distutils import archive_util
from distutils.text_file import TextFile
from distutils.filelist import FileList
from distutils import log
from distutils.util import convert_path
from distutils.errors import DistutilsTemplateError, DistutilsOptionError
def _add_defaults_optional(self):
    optional = ['test/test*.py', 'setup.cfg']
    for pattern in optional:
        files = filter(os.path.isfile, glob(pattern))
        self.filelist.extend(files)