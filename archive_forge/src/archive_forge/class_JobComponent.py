from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class JobComponent(_messages.Message):
    """Message to encapsulate component actuated by a job. JobComponent does
  not represent a GCP API resource.

  Enums:
    OperationValueValuesEnum: Operation to be performed on component.

  Fields:
    operation: Operation to be performed on component.
    typedName: TypedName is the component name and its type.
  """

    class OperationValueValuesEnum(_messages.Enum):
        """Operation to be performed on component.

    Values:
      COMPONENT_OPERATION_UNSPECIFIED: ComponentOperation unset.
      APPLY: Apply configuration to component.
      DESTROY: Destroy component.
    """
        COMPONENT_OPERATION_UNSPECIFIED = 0
        APPLY = 1
        DESTROY = 2
    operation = _messages.EnumField('OperationValueValuesEnum', 1)
    typedName = _messages.MessageField('TypedName', 2)