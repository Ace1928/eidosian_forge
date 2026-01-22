import types
import weakref
from .lock import allocate_lock
from .error import CDefError, VerificationError, VerificationMissing
def _get_c_name(self):
    return self.c_name_with_marker.replace('&', '')