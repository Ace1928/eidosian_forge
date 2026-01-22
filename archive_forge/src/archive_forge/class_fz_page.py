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
class fz_page(object):
    """
    Structure definition is public so other classes can
    derive from it. Do not access the members directly.
    """
    thisown = property(lambda x: x.this.own(), lambda x, v: x.this.own(v), doc='The membership flag')
    __repr__ = _swig_repr
    refs = property(_mupdf.fz_page_refs_get, _mupdf.fz_page_refs_set)
    doc = property(_mupdf.fz_page_doc_get, _mupdf.fz_page_doc_set)
    chapter = property(_mupdf.fz_page_chapter_get, _mupdf.fz_page_chapter_set)
    number = property(_mupdf.fz_page_number_get, _mupdf.fz_page_number_set)
    incomplete = property(_mupdf.fz_page_incomplete_get, _mupdf.fz_page_incomplete_set)
    drop_page = property(_mupdf.fz_page_drop_page_get, _mupdf.fz_page_drop_page_set)
    bound_page = property(_mupdf.fz_page_bound_page_get, _mupdf.fz_page_bound_page_set)
    run_page_contents = property(_mupdf.fz_page_run_page_contents_get, _mupdf.fz_page_run_page_contents_set)
    run_page_annots = property(_mupdf.fz_page_run_page_annots_get, _mupdf.fz_page_run_page_annots_set)
    run_page_widgets = property(_mupdf.fz_page_run_page_widgets_get, _mupdf.fz_page_run_page_widgets_set)
    load_links = property(_mupdf.fz_page_load_links_get, _mupdf.fz_page_load_links_set)
    page_presentation = property(_mupdf.fz_page_page_presentation_get, _mupdf.fz_page_page_presentation_set)
    control_separation = property(_mupdf.fz_page_control_separation_get, _mupdf.fz_page_control_separation_set)
    separation_disabled = property(_mupdf.fz_page_separation_disabled_get, _mupdf.fz_page_separation_disabled_set)
    separations = property(_mupdf.fz_page_separations_get, _mupdf.fz_page_separations_set)
    overprint = property(_mupdf.fz_page_overprint_get, _mupdf.fz_page_overprint_set)
    create_link = property(_mupdf.fz_page_create_link_get, _mupdf.fz_page_create_link_set)
    delete_link = property(_mupdf.fz_page_delete_link_get, _mupdf.fz_page_delete_link_set)
    prev = property(_mupdf.fz_page_prev_get, _mupdf.fz_page_prev_set)
    next = property(_mupdf.fz_page_next_get, _mupdf.fz_page_next_set)

    def __init__(self):
        _mupdf.fz_page_swiginit(self, _mupdf.new_fz_page())
    __swig_destroy__ = _mupdf.delete_fz_page