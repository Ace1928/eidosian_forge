from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GoogleIamV3betaAccessBoundaryPolicyEnforcementVersionStaticVersion(_messages.Message):
    """A specific version number. This will need to be manually updated to
  newer versions as they become available in order to keep maximum protection.

  Fields:
    enforcementVersion: Optional. Currently only a value of '1' is allowed.
  """
    enforcementVersion = _messages.IntegerField(1, variant=_messages.Variant.INT32)