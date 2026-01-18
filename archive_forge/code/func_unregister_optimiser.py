from typing import Generic, List, Type, TypeVar
from .errors import BzrError
from .lock import LogicalLockResult
from .pyutils import get_named_object
@classmethod
def unregister_optimiser(klass, optimiser):
    """Unregister an InterObject optimiser."""
    klass._optimisers.remove(optimiser)