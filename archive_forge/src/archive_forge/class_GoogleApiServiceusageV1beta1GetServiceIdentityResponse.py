from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GoogleApiServiceusageV1beta1GetServiceIdentityResponse(_messages.Message):
    """Response message for getting service identity.

  Enums:
    StateValueValuesEnum: Service identity state.

  Fields:
    identity: Service identity that service producer can use to access
      consumer resources. If exists is true, it contains email and unique_id.
      If exists is false, it contains pre-constructed email and empty
      unique_id.
    state: Service identity state.
  """

    class StateValueValuesEnum(_messages.Enum):
        """Service identity state.

    Values:
      IDENTITY_STATE_UNSPECIFIED: Default service identity state. This value
        is used if the state is omitted.
      ACTIVE: Service identity has been created and can be used.
    """
        IDENTITY_STATE_UNSPECIFIED = 0
        ACTIVE = 1
    identity = _messages.MessageField('GoogleApiServiceusageV1beta1ServiceIdentity', 1)
    state = _messages.EnumField('StateValueValuesEnum', 2)