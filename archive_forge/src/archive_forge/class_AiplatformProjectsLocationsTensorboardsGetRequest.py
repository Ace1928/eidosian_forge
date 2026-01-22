from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class AiplatformProjectsLocationsTensorboardsGetRequest(_messages.Message):
    """A AiplatformProjectsLocationsTensorboardsGetRequest object.

  Fields:
    name: Required. The name of the Tensorboard resource. Format:
      `projects/{project}/locations/{location}/tensorboards/{tensorboard}`
  """
    name = _messages.StringField(1, required=True)