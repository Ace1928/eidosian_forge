from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class AiplatformProjectsLocationsTensorboardsExperimentsRunsGetRequest(_messages.Message):
    """A AiplatformProjectsLocationsTensorboardsExperimentsRunsGetRequest
  object.

  Fields:
    name: Required. The name of the TensorboardRun resource. Format: `projects
      /{project}/locations/{location}/tensorboards/{tensorboard}/experiments/{
      experiment}/runs/{run}`
  """
    name = _messages.StringField(1, required=True)