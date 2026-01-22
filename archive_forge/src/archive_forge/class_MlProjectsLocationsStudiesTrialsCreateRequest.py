from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class MlProjectsLocationsStudiesTrialsCreateRequest(_messages.Message):
    """A MlProjectsLocationsStudiesTrialsCreateRequest object.

  Fields:
    googleCloudMlV1Trial: A GoogleCloudMlV1Trial resource to be passed as the
      request body.
    parent: Required. The name of the study that the trial belongs to.
  """
    googleCloudMlV1Trial = _messages.MessageField('GoogleCloudMlV1Trial', 1)
    parent = _messages.StringField(2, required=True)