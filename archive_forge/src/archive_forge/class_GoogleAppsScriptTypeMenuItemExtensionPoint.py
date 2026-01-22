from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GoogleAppsScriptTypeMenuItemExtensionPoint(_messages.Message):
    """Common format for declaring a menu item or button that appears within a
  host app.

  Fields:
    label: Required. User-visible text that describes the action taken by
      activating this extension point. For example, "Insert invoice."
    logoUrl: The URL for the logo image shown in the add-on toolbar. If not
      set, defaults to the add-on's primary logo URL.
    runFunction: Required. The endpoint to execute when this extension point
      is activated.
  """
    label = _messages.StringField(1)
    logoUrl = _messages.StringField(2)
    runFunction = _messages.StringField(3)