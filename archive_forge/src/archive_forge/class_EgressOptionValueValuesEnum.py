from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class EgressOptionValueValuesEnum(_messages.Enum):
    """Option to configure network egress for the workers.

    Values:
      EGRESS_OPTION_UNSPECIFIED: If set, defaults to PUBLIC_EGRESS.
      NO_PUBLIC_EGRESS: If set, workers are created without any public
        address, which prevents network egress to public IPs unless a network
        proxy is configured.
      PUBLIC_EGRESS: If set, workers are created with a public address which
        allows for public internet egress.
    """
    EGRESS_OPTION_UNSPECIFIED = 0
    NO_PUBLIC_EGRESS = 1
    PUBLIC_EGRESS = 2