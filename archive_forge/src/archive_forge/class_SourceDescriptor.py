from __future__ import absolute_import
import cython
import os
import platform
from unicodedata import normalize
from contextlib import contextmanager
from .. import Utils
from ..Plex.Scanners import Scanner
from ..Plex.Errors import UnrecognizedInput
from .Errors import error, warning, hold_errors, release_errors, CompileError
from .Lexicon import any_string_prefix, make_lexicon, IDENT
from .Future import print_function
class SourceDescriptor(object):
    """
    A SourceDescriptor should be considered immutable.
    """
    filename = None
    in_utility_code = False
    _file_type = 'pyx'
    _escaped_description = None
    _cmp_name = ''

    def __str__(self):
        assert False

    def set_file_type_from_name(self, filename):
        name, ext = os.path.splitext(filename)
        self._file_type = ext in ('.pyx', '.pxd', '.py') and ext[1:] or 'pyx'

    def is_cython_file(self):
        return self._file_type in ('pyx', 'pxd')

    def is_python_file(self):
        return self._file_type == 'py'

    def get_escaped_description(self):
        if self._escaped_description is None:
            esc_desc = self.get_description().encode('ASCII', 'replace').decode('ASCII')
            self._escaped_description = esc_desc.replace('\\', '/')
        return self._escaped_description

    def __gt__(self, other):
        try:
            return self._cmp_name > other._cmp_name
        except AttributeError:
            return False

    def __lt__(self, other):
        try:
            return self._cmp_name < other._cmp_name
        except AttributeError:
            return False

    def __le__(self, other):
        try:
            return self._cmp_name <= other._cmp_name
        except AttributeError:
            return False

    def __copy__(self):
        return self

    def __deepcopy__(self, memo):
        return self