from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
class RoboDirective(_messages.Message):
    """Directs Robo to interact with a specific UI element if it is encountered
  during the crawl. Currently, Robo can perform text entry or element click.

  Enums:
    ActionTypeValueValuesEnum: Required. The type of action that Robo should
      perform on the specified element.

  Fields:
    actionType: Required. The type of action that Robo should perform on the
      specified element.
    inputText: The text that Robo is directed to set. If left empty, the
      directive will be treated as a CLICK on the element matching the
      resource_name.
    resourceName: Required. The android resource name of the target UI
      element. For example, in Java: R.string.foo in xml: @string/foo Only the
      "foo" part is needed. Reference doc:
      https://developer.android.com/guide/topics/resources/accessing-
      resources.html
  """

    class ActionTypeValueValuesEnum(_messages.Enum):
        """Required. The type of action that Robo should perform on the specified
    element.

    Values:
      ACTION_TYPE_UNSPECIFIED: DO NOT USE. For proto versioning only.
      SINGLE_CLICK: Direct Robo to click on the specified element. No-op if
        specified element is not clickable.
      ENTER_TEXT: Direct Robo to enter text on the specified element. No-op if
        specified element is not enabled or does not allow text entry.
      IGNORE: Direct Robo to ignore interactions with a specific element.
    """
        ACTION_TYPE_UNSPECIFIED = 0
        SINGLE_CLICK = 1
        ENTER_TEXT = 2
        IGNORE = 3
    actionType = _messages.EnumField('ActionTypeValueValuesEnum', 1)
    inputText = _messages.StringField(2)
    resourceName = _messages.StringField(3)