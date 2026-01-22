from sys import version_info as _swig_python_version_info
import weakref
import inspect
import os
import re
import sys
import traceback
import inspect
import io
import os
import sys
import traceback
import types
class fz_archive(object):
    """	Implementation details: Subject to change."""
    thisown = property(lambda x: x.this.own(), lambda x, v: x.this.own(v), doc='The membership flag')
    __repr__ = _swig_repr
    refs = property(_mupdf.fz_archive_refs_get, _mupdf.fz_archive_refs_set)
    file = property(_mupdf.fz_archive_file_get, _mupdf.fz_archive_file_set)
    format = property(_mupdf.fz_archive_format_get, _mupdf.fz_archive_format_set)
    drop_archive = property(_mupdf.fz_archive_drop_archive_get, _mupdf.fz_archive_drop_archive_set)
    count_entries = property(_mupdf.fz_archive_count_entries_get, _mupdf.fz_archive_count_entries_set)
    list_entry = property(_mupdf.fz_archive_list_entry_get, _mupdf.fz_archive_list_entry_set)
    has_entry = property(_mupdf.fz_archive_has_entry_get, _mupdf.fz_archive_has_entry_set)
    read_entry = property(_mupdf.fz_archive_read_entry_get, _mupdf.fz_archive_read_entry_set)
    open_entry = property(_mupdf.fz_archive_open_entry_get, _mupdf.fz_archive_open_entry_set)

    def __init__(self):
        _mupdf.fz_archive_swiginit(self, _mupdf.new_fz_archive())
    __swig_destroy__ = _mupdf.delete_fz_archive