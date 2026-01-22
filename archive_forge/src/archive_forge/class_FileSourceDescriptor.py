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
class FileSourceDescriptor(SourceDescriptor):
    """
    Represents a code source. A code source is a more generic abstraction
    for a "filename" (as sometimes the code doesn't come from a file).
    Instances of code sources are passed to Scanner.__init__ as the
    optional name argument and will be passed back when asking for
    the position()-tuple.
    """

    def __init__(self, filename, path_description=None):
        filename = Utils.decode_filename(filename)
        self.path_description = path_description or filename
        self.filename = filename
        workdir = os.path.abspath('.') + os.sep
        self.file_path = filename[len(workdir):] if filename.startswith(workdir) else filename
        self.set_file_type_from_name(filename)
        self._cmp_name = filename
        self._lines = {}

    def get_lines(self, encoding=None, error_handling=None):
        key = (encoding, error_handling)
        try:
            lines = self._lines[key]
            if lines is not None:
                return lines
        except KeyError:
            pass
        with Utils.open_source_file(self.filename, encoding=encoding, error_handling=error_handling) as f:
            lines = list(f)
        if key in self._lines:
            self._lines[key] = lines
        else:
            self._lines[key] = None
        return lines

    def get_description(self):
        try:
            return os.path.relpath(self.path_description)
        except ValueError:
            return self.path_description

    def get_error_description(self):
        path = self.filename
        cwd = Utils.decode_filename(os.getcwd() + os.path.sep)
        if path.startswith(cwd):
            return path[len(cwd):]
        return path

    def get_filenametable_entry(self):
        return self.file_path

    def __eq__(self, other):
        return isinstance(other, FileSourceDescriptor) and self.filename == other.filename

    def __hash__(self):
        return hash(self.filename)

    def __repr__(self):
        return '<FileSourceDescriptor:%s>' % self.filename