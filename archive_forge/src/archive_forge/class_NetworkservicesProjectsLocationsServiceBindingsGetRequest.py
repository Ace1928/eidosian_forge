from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class NetworkservicesProjectsLocationsServiceBindingsGetRequest(_messages.Message):
    """A NetworkservicesProjectsLocationsServiceBindingsGetRequest object.

  Fields:
    name: Required. A name of the ServiceBinding to get. Must be in the format
      `projects/*/locations/global/serviceBindings/*`.
  """
    name = _messages.StringField(1, required=True)