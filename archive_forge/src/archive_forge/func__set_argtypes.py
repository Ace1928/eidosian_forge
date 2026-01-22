import threading
from django.contrib.gis.geos.base import GEOSBase
from django.contrib.gis.geos.libgeos import CONTEXT_PTR, error_h, lgeos, notice_h
def _set_argtypes(self, argtypes):
    self.cfunc.argtypes = [CONTEXT_PTR, *argtypes]