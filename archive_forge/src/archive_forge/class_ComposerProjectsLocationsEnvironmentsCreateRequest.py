from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class ComposerProjectsLocationsEnvironmentsCreateRequest(_messages.Message):
    """A ComposerProjectsLocationsEnvironmentsCreateRequest object.

  Fields:
    environment: A Environment resource to be passed as the request body.
    parent: The parent must be of the form
      "projects/{projectId}/locations/{locationId}".
  """
    environment = _messages.MessageField('Environment', 1)
    parent = _messages.StringField(2, required=True)