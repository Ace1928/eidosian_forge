from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class AiplatformProjectsLocationsTensorboardsExperimentsDeleteRequest(_messages.Message):
    """A AiplatformProjectsLocationsTensorboardsExperimentsDeleteRequest
  object.

  Fields:
    name: Required. The name of the TensorboardExperiment to be deleted.
      Format: `projects/{project}/locations/{location}/tensorboards/{tensorboa
      rd}/experiments/{experiment}`
  """
    name = _messages.StringField(1, required=True)