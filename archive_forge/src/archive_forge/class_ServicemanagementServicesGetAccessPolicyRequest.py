from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class ServicemanagementServicesGetAccessPolicyRequest(_messages.Message):
    """A ServicemanagementServicesGetAccessPolicyRequest object.

  Fields:
    serviceName: The name of the service.  For example:
      `example.googleapis.com`.
  """
    serviceName = _messages.StringField(1, required=True)