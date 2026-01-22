from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class TranscoderProjectsLocationsJobTemplatesCreateRequest(_messages.Message):
    """A TranscoderProjectsLocationsJobTemplatesCreateRequest object.

  Fields:
    jobTemplate: A JobTemplate resource to be passed as the request body.
    jobTemplateId: Required. The ID to use for the job template, which will
      become the final component of the job template's resource name. This
      value should be 4-63 characters, and valid characters must match the
      regular expression `a-zA-Z*`.
    parent: Required. The parent location to create this job template. Format:
      `projects/{project}/locations/{location}`
  """
    jobTemplate = _messages.MessageField('JobTemplate', 1)
    jobTemplateId = _messages.StringField(2)
    parent = _messages.StringField(3, required=True)