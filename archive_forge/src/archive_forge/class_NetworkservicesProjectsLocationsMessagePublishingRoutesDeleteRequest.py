from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class NetworkservicesProjectsLocationsMessagePublishingRoutesDeleteRequest(_messages.Message):
    """A NetworkservicesProjectsLocationsMessagePublishingRoutesDeleteRequest
  object.

  Fields:
    name: Required. A name of the MessagePublishingRoute to delete. Must be in
      the format `projects/*/locations/*/messagePublishingRoutes/*`.
  """
    name = _messages.StringField(1, required=True)