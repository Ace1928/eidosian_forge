from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class AiplatformProjectsLocationsModelsDeleteVersionRequest(_messages.Message):
    """A AiplatformProjectsLocationsModelsDeleteVersionRequest object.

  Fields:
    name: Required. The name of the model version to be deleted, with a
      version ID explicitly included. Example:
      `projects/{project}/locations/{location}/models/{model}@1234`
  """
    name = _messages.StringField(1, required=True)