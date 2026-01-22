from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class ServicemanagementServicesConfigsGetRequest(_messages.Message):
    """A ServicemanagementServicesConfigsGetRequest object.

  Fields:
    configId: The id of the service config resource. Optional. If it is not
      specified, the latest version of config will be returned.
    serviceName: The name of the service.  See the `ServiceManager` overview
      for naming requirements.  For example: `example.googleapis.com`.
  """
    configId = _messages.StringField(1, required=True)
    serviceName = _messages.StringField(2, required=True)