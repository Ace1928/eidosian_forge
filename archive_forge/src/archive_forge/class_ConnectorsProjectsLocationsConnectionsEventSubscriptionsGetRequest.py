from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class ConnectorsProjectsLocationsConnectionsEventSubscriptionsGetRequest(_messages.Message):
    """A ConnectorsProjectsLocationsConnectionsEventSubscriptionsGetRequest
  object.

  Fields:
    name: Required. Resource name of the form:
      `projects/*/locations/*/connections/*/eventSubscriptions/*`
  """
    name = _messages.StringField(1, required=True)