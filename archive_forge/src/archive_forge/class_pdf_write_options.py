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
class pdf_write_options(object):
    thisown = property(lambda x: x.this.own(), lambda x, v: x.this.own(v), doc='The membership flag')
    __repr__ = _swig_repr
    do_incremental = property(_mupdf.pdf_write_options_do_incremental_get, _mupdf.pdf_write_options_do_incremental_set)
    do_pretty = property(_mupdf.pdf_write_options_do_pretty_get, _mupdf.pdf_write_options_do_pretty_set)
    do_ascii = property(_mupdf.pdf_write_options_do_ascii_get, _mupdf.pdf_write_options_do_ascii_set)
    do_compress = property(_mupdf.pdf_write_options_do_compress_get, _mupdf.pdf_write_options_do_compress_set)
    do_compress_images = property(_mupdf.pdf_write_options_do_compress_images_get, _mupdf.pdf_write_options_do_compress_images_set)
    do_compress_fonts = property(_mupdf.pdf_write_options_do_compress_fonts_get, _mupdf.pdf_write_options_do_compress_fonts_set)
    do_decompress = property(_mupdf.pdf_write_options_do_decompress_get, _mupdf.pdf_write_options_do_decompress_set)
    do_garbage = property(_mupdf.pdf_write_options_do_garbage_get, _mupdf.pdf_write_options_do_garbage_set)
    do_linear = property(_mupdf.pdf_write_options_do_linear_get, _mupdf.pdf_write_options_do_linear_set)
    do_clean = property(_mupdf.pdf_write_options_do_clean_get, _mupdf.pdf_write_options_do_clean_set)
    do_sanitize = property(_mupdf.pdf_write_options_do_sanitize_get, _mupdf.pdf_write_options_do_sanitize_set)
    do_appearance = property(_mupdf.pdf_write_options_do_appearance_get, _mupdf.pdf_write_options_do_appearance_set)
    do_encrypt = property(_mupdf.pdf_write_options_do_encrypt_get, _mupdf.pdf_write_options_do_encrypt_set)
    dont_regenerate_id = property(_mupdf.pdf_write_options_dont_regenerate_id_get, _mupdf.pdf_write_options_dont_regenerate_id_set)
    permissions = property(_mupdf.pdf_write_options_permissions_get, _mupdf.pdf_write_options_permissions_set)
    opwd_utf8 = property(_mupdf.pdf_write_options_opwd_utf8_get, _mupdf.pdf_write_options_opwd_utf8_set)
    upwd_utf8 = property(_mupdf.pdf_write_options_upwd_utf8_get, _mupdf.pdf_write_options_upwd_utf8_set)
    do_snapshot = property(_mupdf.pdf_write_options_do_snapshot_get, _mupdf.pdf_write_options_do_snapshot_set)
    do_preserve_metadata = property(_mupdf.pdf_write_options_do_preserve_metadata_get, _mupdf.pdf_write_options_do_preserve_metadata_set)
    do_use_objstms = property(_mupdf.pdf_write_options_do_use_objstms_get, _mupdf.pdf_write_options_do_use_objstms_set)
    compression_effort = property(_mupdf.pdf_write_options_compression_effort_get, _mupdf.pdf_write_options_compression_effort_set)

    def __init__(self):
        _mupdf.pdf_write_options_swiginit(self, _mupdf.new_pdf_write_options())
    __swig_destroy__ = _mupdf.delete_pdf_write_options