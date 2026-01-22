from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class MlProjectsLocationsStudiesCreateRequest(_messages.Message):
    """A MlProjectsLocationsStudiesCreateRequest object.

  Fields:
    googleCloudMlV1Study: A GoogleCloudMlV1Study resource to be passed as the
      request body.
    parent: Required. The project and location that the study belongs to.
      Format: projects/{project}/locations/{location}
    studyId: Required. The ID to use for the study, which will become the
      final component of the study's resource name.
  """
    googleCloudMlV1Study = _messages.MessageField('GoogleCloudMlV1Study', 1)
    parent = _messages.StringField(2, required=True)
    studyId = _messages.StringField(3)