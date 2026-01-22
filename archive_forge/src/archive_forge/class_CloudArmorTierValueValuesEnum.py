from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class CloudArmorTierValueValuesEnum(_messages.Enum):
    """Managed protection tier to be set.

    Values:
      CA_ENTERPRISE_ANNUAL: Enterprise tier protection billed annually.
      CA_ENTERPRISE_PAYGO: Enterprise tier protection billed monthly.
      CA_STANDARD: Standard protection.
    """
    CA_ENTERPRISE_ANNUAL = 0
    CA_ENTERPRISE_PAYGO = 1
    CA_STANDARD = 2