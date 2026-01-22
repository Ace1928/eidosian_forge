from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class ApphubProjectsLocationsServiceProjectAttachmentsGetRequest(_messages.Message):
    """A ApphubProjectsLocationsServiceProjectAttachmentsGetRequest object.

  Fields:
    name: Required. Fully qualified name of the service project attachment to
      retrieve. Expected format: `projects/{project}/locations/{location}/serv
      iceProjectAttachments/{serviceProjectAttachment}`.
  """
    name = _messages.StringField(1, required=True)