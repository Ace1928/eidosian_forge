from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class AiplatformProjectsLocationsTensorboardsExperimentsRunsDeleteRequest(_messages.Message):
    """A AiplatformProjectsLocationsTensorboardsExperimentsRunsDeleteRequest
  object.

  Fields:
    name: Required. The name of the TensorboardRun to be deleted. Format: `pro
      jects/{project}/locations/{location}/tensorboards/{tensorboard}/experime
      nts/{experiment}/runs/{run}`
  """
    name = _messages.StringField(1, required=True)