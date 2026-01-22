from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class PrivilegedAccess(_messages.Message):
    """Privileged access that this service can be used to gate.

  Fields:
    gcpIamAccess: Access to a GCP resource through IAM.
  """
    gcpIamAccess = _messages.MessageField('GcpIamAccess', 1)