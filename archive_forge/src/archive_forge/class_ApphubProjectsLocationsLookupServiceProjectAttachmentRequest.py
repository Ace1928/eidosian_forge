from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class ApphubProjectsLocationsLookupServiceProjectAttachmentRequest(_messages.Message):
    """A ApphubProjectsLocationsLookupServiceProjectAttachmentRequest object.

  Fields:
    name: Required. Service project ID and location to lookup service project
      attachment for. Only global location is supported. Expected format:
      `projects/{project}/locations/{location}`.
  """
    name = _messages.StringField(1, required=True)