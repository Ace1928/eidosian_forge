import threading
from django.contrib.gis.geos.base import GEOSBase
from django.contrib.gis.geos.libgeos import CONTEXT_PTR, error_h, lgeos, notice_h

    Serve as a wrapper for GEOS C Functions. Use thread-safe function
    variants when available.
    