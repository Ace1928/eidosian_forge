from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class AiplatformProjectsLocationsStudiesGetRequest(_messages.Message):
    """A AiplatformProjectsLocationsStudiesGetRequest object.

  Fields:
    name: Required. The name of the Study resource. Format:
      `projects/{project}/locations/{location}/studies/{study}`
  """
    name = _messages.StringField(1, required=True)