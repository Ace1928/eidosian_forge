from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GoogleAppsScriptTypeGmailUniversalAction(_messages.Message):
    """An action that is always available in the add-on toolbar menu regardless
  of message context.

  Fields:
    openLink: A link that is opened by Gmail when the user triggers the
      action.
    runFunction: An endpoint that is called when the user triggers the action.
      See the [universal actions guide](/gmail/add-ons/how-tos/universal-
      actions) for details.
    text: Required. User-visible text describing the action, for example, "Add
      a new contact."
  """
    openLink = _messages.StringField(1)
    runFunction = _messages.StringField(2)
    text = _messages.StringField(3)