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
class pdf_cmap(object):
    thisown = property(lambda x: x.this.own(), lambda x, v: x.this.own(v), doc='The membership flag')
    __repr__ = _swig_repr
    storable = property(_mupdf.pdf_cmap_storable_get, _mupdf.pdf_cmap_storable_set)
    cmap_name = property(_mupdf.pdf_cmap_cmap_name_get, _mupdf.pdf_cmap_cmap_name_set)
    usecmap_name = property(_mupdf.pdf_cmap_usecmap_name_get, _mupdf.pdf_cmap_usecmap_name_set)
    usecmap = property(_mupdf.pdf_cmap_usecmap_get, _mupdf.pdf_cmap_usecmap_set)
    wmode = property(_mupdf.pdf_cmap_wmode_get, _mupdf.pdf_cmap_wmode_set)
    codespace_len = property(_mupdf.pdf_cmap_codespace_len_get, _mupdf.pdf_cmap_codespace_len_set)
    rlen = property(_mupdf.pdf_cmap_rlen_get, _mupdf.pdf_cmap_rlen_set)
    rcap = property(_mupdf.pdf_cmap_rcap_get, _mupdf.pdf_cmap_rcap_set)
    ranges = property(_mupdf.pdf_cmap_ranges_get, _mupdf.pdf_cmap_ranges_set)
    xlen = property(_mupdf.pdf_cmap_xlen_get, _mupdf.pdf_cmap_xlen_set)
    xcap = property(_mupdf.pdf_cmap_xcap_get, _mupdf.pdf_cmap_xcap_set)
    xranges = property(_mupdf.pdf_cmap_xranges_get, _mupdf.pdf_cmap_xranges_set)
    mlen = property(_mupdf.pdf_cmap_mlen_get, _mupdf.pdf_cmap_mlen_set)
    mcap = property(_mupdf.pdf_cmap_mcap_get, _mupdf.pdf_cmap_mcap_set)
    mranges = property(_mupdf.pdf_cmap_mranges_get, _mupdf.pdf_cmap_mranges_set)
    dlen = property(_mupdf.pdf_cmap_dlen_get, _mupdf.pdf_cmap_dlen_set)
    dcap = property(_mupdf.pdf_cmap_dcap_get, _mupdf.pdf_cmap_dcap_set)
    dict = property(_mupdf.pdf_cmap_dict_get, _mupdf.pdf_cmap_dict_set)
    tlen = property(_mupdf.pdf_cmap_tlen_get, _mupdf.pdf_cmap_tlen_set)
    tcap = property(_mupdf.pdf_cmap_tcap_get, _mupdf.pdf_cmap_tcap_set)
    ttop = property(_mupdf.pdf_cmap_ttop_get, _mupdf.pdf_cmap_ttop_set)
    tree = property(_mupdf.pdf_cmap_tree_get, _mupdf.pdf_cmap_tree_set)

    def __init__(self):
        _mupdf.pdf_cmap_swiginit(self, _mupdf.new_pdf_cmap())
    __swig_destroy__ = _mupdf.delete_pdf_cmap