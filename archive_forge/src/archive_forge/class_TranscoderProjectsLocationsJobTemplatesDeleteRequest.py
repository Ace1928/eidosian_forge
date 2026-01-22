from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class TranscoderProjectsLocationsJobTemplatesDeleteRequest(_messages.Message):
    """A TranscoderProjectsLocationsJobTemplatesDeleteRequest object.

  Fields:
    allowMissing: If set to true, and the job template is not found, the
      request will succeed but no action will be taken on the server.
    name: Required. The name of the job template to delete.
      `projects/{project}/locations/{location}/jobTemplates/{job_template}`
  """
    allowMissing = _messages.BooleanField(1)
    name = _messages.StringField(2, required=True)