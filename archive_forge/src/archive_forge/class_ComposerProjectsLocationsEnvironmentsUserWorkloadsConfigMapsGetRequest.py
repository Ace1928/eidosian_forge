from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class ComposerProjectsLocationsEnvironmentsUserWorkloadsConfigMapsGetRequest(_messages.Message):
    """A ComposerProjectsLocationsEnvironmentsUserWorkloadsConfigMapsGetRequest
  object.

  Fields:
    name: Required. The resource name of the ConfigMap to get, in the form: "p
      rojects/{projectId}/locations/{locationId}/environments/{environmentId}/
      userWorkloadsConfigMaps/{userWorkloadsConfigMapId}"
  """
    name = _messages.StringField(1, required=True)