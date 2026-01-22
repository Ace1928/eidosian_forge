from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class ComposerProjectsLocationsEnvironmentsGetRequest(_messages.Message):
    """A ComposerProjectsLocationsEnvironmentsGetRequest object.

  Fields:
    name: The resource name of the environment to get, in the form:
      "projects/{projectId}/locations/{locationId}/environments/{environmentId
      }"
  """
    name = _messages.StringField(1, required=True)