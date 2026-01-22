from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class CustomDomain(_messages.Message):
    """Custom domain information.

  Enums:
    StateValueValuesEnum: Domain state.

  Fields:
    domain: Domain name.
    state: Domain state.
  """

    class StateValueValuesEnum(_messages.Enum):
        """Domain state.

    Values:
      CUSTOM_DOMAIN_STATE_UNSPECIFIED: Unspecified state.
      UNVERIFIED: DNS record is not created.
      VERIFIED: DNS record is created.
      MODIFYING: Calling SLM to update.
      AVAILABLE: ManagedCertificate is ready.
      UNAVAILABLE: ManagedCertificate is not ready.
      UNKNOWN: Status is not known.
    """
        CUSTOM_DOMAIN_STATE_UNSPECIFIED = 0
        UNVERIFIED = 1
        VERIFIED = 2
        MODIFYING = 3
        AVAILABLE = 4
        UNAVAILABLE = 5
        UNKNOWN = 6
    domain = _messages.StringField(1)
    state = _messages.EnumField('StateValueValuesEnum', 2)