from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class AiplatformProjectsLocationsModelsGetRequest(_messages.Message):
    """A AiplatformProjectsLocationsModelsGetRequest object.

  Fields:
    name: Required. The name of the Model resource. Format:
      `projects/{project}/locations/{location}/models/{model}` In order to
      retrieve a specific version of the model, also provide the version ID or
      version alias. Example:
      `projects/{project}/locations/{location}/models/{model}@2` or
      `projects/{project}/locations/{location}/models/{model}@golden` If no
      version ID or alias is specified, the "default" version will be
      returned. The "default" version alias is created for the first version
      of the model, and can be moved to other versions later on. There will be
      exactly one default version.
  """
    name = _messages.StringField(1, required=True)