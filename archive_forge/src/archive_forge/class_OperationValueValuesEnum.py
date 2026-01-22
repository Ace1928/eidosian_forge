from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class OperationValueValuesEnum(_messages.Enum):
    """Operation to be performed on the resource .

    Values:
      OPERATION_UNSPECIFIED: Default value indicating the operation is
        unknown.
      APPLY: Apply configuration to resource.
      DESTROY: Destroy resource.
    """
    OPERATION_UNSPECIFIED = 0
    APPLY = 1
    DESTROY = 2