from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GoogleAppsScriptTypeGmailComposeTrigger(_messages.Message):
    """A trigger that activates when user is composing an email.

  Enums:
    DraftAccessValueValuesEnum: Defines the level of data access when a
      compose time add-on is triggered.

  Fields:
    actions: Defines the set of actions for a compose time add-on. These are
      actions that users can trigger on a compose time add-on.
    draftAccess: Defines the level of data access when a compose time add-on
      is triggered.
  """

    class DraftAccessValueValuesEnum(_messages.Enum):
        """Defines the level of data access when a compose time add-on is
    triggered.

    Values:
      UNSPECIFIED: Default value when nothing is set for draftAccess.
      NONE: The compose trigger can't access any data of the draft when a
        compose add-on is triggered.
      METADATA: Gives the compose trigger the permission to access the
        metadata of the draft when a compose add-on is triggered. This
        includes the audience list, such as the To and Cc list of a draft
        message.
    """
        UNSPECIFIED = 0
        NONE = 1
        METADATA = 2
    actions = _messages.MessageField('GoogleAppsScriptTypeMenuItemExtensionPoint', 1, repeated=True)
    draftAccess = _messages.EnumField('DraftAccessValueValuesEnum', 2)