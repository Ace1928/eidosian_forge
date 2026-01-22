from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class ServicemanagementServicesConfigsListRequest(_messages.Message):
    """A ServicemanagementServicesConfigsListRequest object.

  Fields:
    pageSize: The max number of items to include in the response list.
    pageToken: The token of the page to retrieve.
    serviceName: The name of the service.  See the `ServiceManager` overview
      for naming requirements.  For example: `example.googleapis.com`.
  """
    pageSize = _messages.IntegerField(1, variant=_messages.Variant.INT32)
    pageToken = _messages.StringField(2)
    serviceName = _messages.StringField(3, required=True)