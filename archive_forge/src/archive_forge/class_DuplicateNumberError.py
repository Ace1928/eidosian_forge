import types
import weakref
import six
from apitools.base.protorpclite import util
class DuplicateNumberError(Error):
    """Duplicate number assigned to field."""