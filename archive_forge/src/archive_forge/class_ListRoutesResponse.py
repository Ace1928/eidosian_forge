from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class ListRoutesResponse(_messages.Message):
    """ListRoutesResponse is a list of Route resources.

  Fields:
    apiVersion: The API version for this call such as
      "serving.knative.dev/v1".
    items: List of Routes.
    kind: The kind of this resource, in this case always "RouteList".
    metadata: Metadata associated with this Route list.
    unreachable: Locations that could not be reached.
  """
    apiVersion = _messages.StringField(1)
    items = _messages.MessageField('Route', 2, repeated=True)
    kind = _messages.StringField(3)
    metadata = _messages.MessageField('ListMeta', 4)
    unreachable = _messages.StringField(5, repeated=True)