from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class TranscoderProjectsLocationsJobTemplatesGetRequest(_messages.Message):
    """A TranscoderProjectsLocationsJobTemplatesGetRequest object.

  Fields:
    name: Required. The name of the job template to retrieve. Format:
      `projects/{project}/locations/{location}/jobTemplates/{job_template}`
  """
    name = _messages.StringField(1, required=True)