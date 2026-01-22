from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class TrustModelValueValuesEnum(_messages.Enum):
    """Trust Model of the SSL connection

    Values:
      PUBLIC: Public Trust Model. Takes the Default Java trust store.
      PRIVATE: Private Trust Model. Takes custom/private trust store.
      INSECURE: Insecure Trust Model. Accept all certificates.
    """
    PUBLIC = 0
    PRIVATE = 1
    INSECURE = 2