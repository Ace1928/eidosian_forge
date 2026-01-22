import types
import weakref
from .lock import allocate_lock
from .error import CDefError, VerificationError, VerificationMissing
class BasePrimitiveType(BaseType):

    def is_complex_type(self):
        return False