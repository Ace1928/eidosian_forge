from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class ComposerProjectsLocationsEnvironmentsDagsPauseRequest(_messages.Message):
    """A ComposerProjectsLocationsEnvironmentsDagsPauseRequest object.

  Fields:
    name: Required. The name of dag to pause.
    pauseDagRequest: A PauseDagRequest resource to be passed as the request
      body.
  """
    name = _messages.StringField(1, required=True)
    pauseDagRequest = _messages.MessageField('PauseDagRequest', 2)