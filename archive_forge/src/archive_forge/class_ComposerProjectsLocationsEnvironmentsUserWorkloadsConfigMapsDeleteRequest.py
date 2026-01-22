from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class ComposerProjectsLocationsEnvironmentsUserWorkloadsConfigMapsDeleteRequest(_messages.Message):
    """A
  ComposerProjectsLocationsEnvironmentsUserWorkloadsConfigMapsDeleteRequest
  object.

  Fields:
    name: Required. The ConfigMap to delete, in the form: "projects/{projectId
      }/locations/{locationId}/environments/{environmentId}/userWorkloadsConfi
      gMaps/{userWorkloadsConfigMapId}"
  """
    name = _messages.StringField(1, required=True)