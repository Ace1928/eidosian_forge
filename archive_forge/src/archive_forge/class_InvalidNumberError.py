import types
import weakref
import six
from apitools.base.protorpclite import util
class InvalidNumberError(FieldDefinitionError):
    """Invalid number provided to field."""