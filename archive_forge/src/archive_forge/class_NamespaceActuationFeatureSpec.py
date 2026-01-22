from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class NamespaceActuationFeatureSpec(_messages.Message):
    """An empty spec for actuation feature. This is required since Feature
  proto requires a spec.

  Enums:
    ActuationModeValueValuesEnum: actuation_mode controls the behavior of the
      controller

  Fields:
    actuationMode: actuation_mode controls the behavior of the controller
  """

    class ActuationModeValueValuesEnum(_messages.Enum):
        """actuation_mode controls the behavior of the controller

    Values:
      ACTUATION_MODE_UNSPECIFIED: ACTUATION_MODE_UNSPECIFIED is similar to
        CREATE_AND_DELETE_IF_CREATED in the default controller behavior.
      ACTUATION_MODE_CREATE_AND_DELETE_IF_CREATED:
        ACTUATION_MODE_CREATE_AND_DELETE_IF_CREATED has the controller create
        cluster namespaces for each fleet namespace and it deletes only the
        ones it created, which are identified by a label.
      ACTUATION_MODE_ADD_AND_REMOVE_FLEET_LABELS:
        ACTUATION_MODE_ADD_AND_REMOVE_FLEET_LABELS has the controller only
        apply labels to cluster namespaces to signal fleet namespace
        enablement. It doesn't create or delete cluster namespaces.
    """
        ACTUATION_MODE_UNSPECIFIED = 0
        ACTUATION_MODE_CREATE_AND_DELETE_IF_CREATED = 1
        ACTUATION_MODE_ADD_AND_REMOVE_FLEET_LABELS = 2
    actuationMode = _messages.EnumField('ActuationModeValueValuesEnum', 1)