from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class CloudEndpoints(_messages.Message):
    """Cloud Endpoints service. Learn more at
  https://cloud.google.com/endpoints.

  Fields:
    service: The name of the Cloud Endpoints service underlying this service.
      Corresponds to the service resource label in the api monitored resource
      (https://cloud.google.com/monitoring/api/resources#tag_api).
  """
    service = _messages.StringField(1)