from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GoogleAppsScriptTypeUniversalActionExtensionPoint(_messages.Message):
    """Format for declaring a universal action menu item extension point.

  Fields:
    label: Required. User-visible text that describes the action taken by
      activating this extension point, for example, "Add a new contact."
    openLink: URL to be opened by the UniversalAction.
    runFunction: Endpoint to be run by the UniversalAction.
  """
    label = _messages.StringField(1)
    openLink = _messages.StringField(2)
    runFunction = _messages.StringField(3)