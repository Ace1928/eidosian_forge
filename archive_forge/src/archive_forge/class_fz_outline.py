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
class fz_outline(object):
    """
    Structure based API
    fz_outline is a tree of the outline of a document (also known
    as table of contents).

    title: Title of outline item using UTF-8 encoding. May be NULL
    if the outline item has no text string.

    uri: Destination in the document to be displayed when this
    outline item is activated. May be an internal or external
    link, or NULL if the outline item does not have a destination.

    page: The page number of an internal link, or -1 for external
    links or links with no destination.

    next: The next outline item at the same level as this outline
    item. May be NULL if no more outline items exist at this level.

    down: The outline items immediate children in the hierarchy.
    May be NULL if no children exist.
    """
    thisown = property(lambda x: x.this.own(), lambda x, v: x.this.own(v), doc='The membership flag')
    __repr__ = _swig_repr
    refs = property(_mupdf.fz_outline_refs_get, _mupdf.fz_outline_refs_set)
    title = property(_mupdf.fz_outline_title_get, _mupdf.fz_outline_title_set)
    uri = property(_mupdf.fz_outline_uri_get, _mupdf.fz_outline_uri_set)
    page = property(_mupdf.fz_outline_page_get, _mupdf.fz_outline_page_set)
    x = property(_mupdf.fz_outline_x_get, _mupdf.fz_outline_x_set)
    y = property(_mupdf.fz_outline_y_get, _mupdf.fz_outline_y_set)
    next = property(_mupdf.fz_outline_next_get, _mupdf.fz_outline_next_set)
    down = property(_mupdf.fz_outline_down_get, _mupdf.fz_outline_down_set)
    is_open = property(_mupdf.fz_outline_is_open_get, _mupdf.fz_outline_is_open_set)

    def __init__(self):
        _mupdf.fz_outline_swiginit(self, _mupdf.new_fz_outline())
    __swig_destroy__ = _mupdf.delete_fz_outline