from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class ComposerProjectsLocationsEnvironmentsDeleteRequest(_messages.Message):
    """A ComposerProjectsLocationsEnvironmentsDeleteRequest object.

  Fields:
    name: The environment to delete, in the form: "projects/{projectId}/locati
      ons/{locationId}/environments/{environmentId}".
  """
    name = _messages.StringField(1, required=True)