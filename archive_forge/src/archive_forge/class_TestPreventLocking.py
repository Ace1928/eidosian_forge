from .. import errors
from ..decorators import only_raises
class TestPreventLocking(errors.LockError):
    """A test exception for forcing locking failure: %(message)s"""