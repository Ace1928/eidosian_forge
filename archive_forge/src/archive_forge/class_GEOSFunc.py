import threading
from django.contrib.gis.geos.base import GEOSBase
from django.contrib.gis.geos.libgeos import CONTEXT_PTR, error_h, lgeos, notice_h
class GEOSFunc:
    """
    Serve as a wrapper for GEOS C Functions. Use thread-safe function
    variants when available.
    """

    def __init__(self, func_name):
        self.cfunc = getattr(lgeos, func_name + '_r')
        self.thread_context = thread_context

    def __call__(self, *args):
        self.thread_context.handle = self.thread_context.handle or GEOSContextHandle()
        return self.cfunc(self.thread_context.handle.ptr, *args)

    def __str__(self):
        return self.cfunc.__name__

    def _get_argtypes(self):
        return self.cfunc.argtypes

    def _set_argtypes(self, argtypes):
        self.cfunc.argtypes = [CONTEXT_PTR, *argtypes]
    argtypes = property(_get_argtypes, _set_argtypes)

    def _get_restype(self):
        return self.cfunc.restype

    def _set_restype(self, restype):
        self.cfunc.restype = restype
    restype = property(_get_restype, _set_restype)

    def _get_errcheck(self):
        return self.cfunc.errcheck

    def _set_errcheck(self, errcheck):
        self.cfunc.errcheck = errcheck
    errcheck = property(_get_errcheck, _set_errcheck)