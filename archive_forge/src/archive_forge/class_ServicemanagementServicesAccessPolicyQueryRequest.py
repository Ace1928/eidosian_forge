from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class ServicemanagementServicesAccessPolicyQueryRequest(_messages.Message):
    """A ServicemanagementServicesAccessPolicyQueryRequest object.

  Fields:
    serviceName: The service to query access for.
    userEmail: The user to query access for.
  """
    serviceName = _messages.StringField(1, required=True)
    userEmail = _messages.StringField(2)