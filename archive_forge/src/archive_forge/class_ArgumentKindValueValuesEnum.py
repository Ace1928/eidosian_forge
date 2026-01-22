from __future__ import absolute_import
from apitools.base.protorpclite import message_types as _message_types
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class ArgumentKindValueValuesEnum(_messages.Enum):
    """Optional. Defaults to FIXED_TYPE.

    Values:
      ARGUMENT_KIND_UNSPECIFIED: Default value.
      FIXED_TYPE: The argument is a variable with fully specified type, which
        can be a struct or an array, but not a table.
      ANY_TYPE: The argument is any type, including struct or array, but not a
        table. To be added: FIXED_TABLE, ANY_TABLE
    """
    ARGUMENT_KIND_UNSPECIFIED = 0
    FIXED_TYPE = 1
    ANY_TYPE = 2