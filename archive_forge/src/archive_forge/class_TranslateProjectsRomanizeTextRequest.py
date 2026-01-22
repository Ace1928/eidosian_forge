from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class TranslateProjectsRomanizeTextRequest(_messages.Message):
    """A TranslateProjectsRomanizeTextRequest object.

  Fields:
    parent: Required. Project or location to make a call. Must refer to a
      caller's project. Format: `projects/{project-number-or-
      id}/locations/{location-id}` or `projects/{project-number-or-id}`. For
      global calls, use `projects/{project-number-or-id}/locations/global` or
      `projects/{project-number-or-id}`.
    romanizeTextRequest: A RomanizeTextRequest resource to be passed as the
      request body.
  """
    parent = _messages.StringField(1, required=True)
    romanizeTextRequest = _messages.MessageField('RomanizeTextRequest', 2)