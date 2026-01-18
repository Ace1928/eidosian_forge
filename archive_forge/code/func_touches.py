from .base import GEOSBase
from .prototypes import prepared as capi
def touches(self, other):
    return capi.prepared_touches(self.ptr, other.ptr)