from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class NetworkservicesProjectsLocationsMulticastGroupConsumerActivationsGetRequest(_messages.Message):
    """A
  NetworkservicesProjectsLocationsMulticastGroupConsumerActivationsGetRequest
  object.

  Fields:
    name: Required. The resource name of the multicast group consumer
      activation to get. Use the following format:
      `projects/*/locations/*/multicastGroupConsumerActivations/*`.
  """
    name = _messages.StringField(1, required=True)