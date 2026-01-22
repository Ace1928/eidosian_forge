from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class SpeechProjectsLocationsCustomClassesDeleteRequest(_messages.Message):
    """A SpeechProjectsLocationsCustomClassesDeleteRequest object.

  Fields:
    allowMissing: If set to true, and the CustomClass is not found, the
      request will succeed and be a no-op (no Operation is recorded in this
      case).
    etag: This checksum is computed by the server based on the value of other
      fields. This may be sent on update, undelete, and delete requests to
      ensure the client has an up-to-date value before proceeding.
    name: Required. The name of the CustomClass to delete. Format:
      `projects/{project}/locations/{location}/customClasses/{custom_class}`
    validateOnly: If set, validate the request and preview the deleted
      CustomClass, but do not actually delete it.
  """
    allowMissing = _messages.BooleanField(1)
    etag = _messages.StringField(2)
    name = _messages.StringField(3, required=True)
    validateOnly = _messages.BooleanField(4)