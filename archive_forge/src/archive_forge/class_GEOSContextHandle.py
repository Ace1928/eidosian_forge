import threading
from django.contrib.gis.geos.base import GEOSBase
from django.contrib.gis.geos.libgeos import CONTEXT_PTR, error_h, lgeos, notice_h
class GEOSContextHandle(GEOSBase):
    """Represent a GEOS context handle."""
    ptr_type = CONTEXT_PTR
    destructor = lgeos.finishGEOS_r

    def __init__(self):
        self.ptr = lgeos.initGEOS_r(notice_h, error_h)