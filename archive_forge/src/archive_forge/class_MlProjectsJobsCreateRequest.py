from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class MlProjectsJobsCreateRequest(_messages.Message):
    """A MlProjectsJobsCreateRequest object.

  Fields:
    googleCloudMlV1Job: A GoogleCloudMlV1Job resource to be passed as the
      request body.
    parent: Required. The project name.
  """
    googleCloudMlV1Job = _messages.MessageField('GoogleCloudMlV1Job', 1)
    parent = _messages.StringField(2, required=True)