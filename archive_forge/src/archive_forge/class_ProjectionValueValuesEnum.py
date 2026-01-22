from apitools.base.protorpclite import message_types as _message_types
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class ProjectionValueValuesEnum(_messages.Enum):
    """Set of properties to return. Defaults to noAcl.

    Values:
      full: Include all properties.
      noAcl: Omit the owner, acl property.
    """
    full = 0
    noAcl = 1