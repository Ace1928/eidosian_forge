from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class ComposerProjectsLocationsEnvironmentsDagsActivateRequest(_messages.Message):
    """A ComposerProjectsLocationsEnvironmentsDagsActivateRequest object.

  Fields:
    activateDagRequest: A ActivateDagRequest resource to be passed as the
      request body.
    name: Required. The name of dag to pause.
  """
    activateDagRequest = _messages.MessageField('ActivateDagRequest', 1)
    name = _messages.StringField(2, required=True)