from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class NetworkservicesProjectsLocationsMulticastGroupDefinitionsGetRequest(_messages.Message):
    """A NetworkservicesProjectsLocationsMulticastGroupDefinitionsGetRequest
  object.

  Fields:
    name: Required. The resource name of the multicast group definition to
      get. Use the following format:
      `projects/*/locations/global/multicastGroupDefinitions/*`.
  """
    name = _messages.StringField(1, required=True)