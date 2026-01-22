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
class pdf_pkcs7_signer(object):
    thisown = property(lambda x: x.this.own(), lambda x, v: x.this.own(v), doc='The membership flag')
    __repr__ = _swig_repr
    keep = property(_mupdf.pdf_pkcs7_signer_keep_get, _mupdf.pdf_pkcs7_signer_keep_set)
    drop = property(_mupdf.pdf_pkcs7_signer_drop_get, _mupdf.pdf_pkcs7_signer_drop_set)
    get_signing_name = property(_mupdf.pdf_pkcs7_signer_get_signing_name_get, _mupdf.pdf_pkcs7_signer_get_signing_name_set)
    max_digest_size = property(_mupdf.pdf_pkcs7_signer_max_digest_size_get, _mupdf.pdf_pkcs7_signer_max_digest_size_set)
    create_digest = property(_mupdf.pdf_pkcs7_signer_create_digest_get, _mupdf.pdf_pkcs7_signer_create_digest_set)

    def __init__(self):
        _mupdf.pdf_pkcs7_signer_swiginit(self, _mupdf.new_pdf_pkcs7_signer())
    __swig_destroy__ = _mupdf.delete_pdf_pkcs7_signer