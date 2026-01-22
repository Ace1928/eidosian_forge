from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GoogleCloudDialogflowV2IntentMessageBrowseCarouselCardBrowseCarouselCardItemOpenUrlAction(_messages.Message):
    """Actions on Google action to open a given url.

  Enums:
    UrlTypeHintValueValuesEnum: Optional. Specifies the type of viewer that is
      used when opening the URL. Defaults to opening via web browser.

  Fields:
    url: Required. URL
    urlTypeHint: Optional. Specifies the type of viewer that is used when
      opening the URL. Defaults to opening via web browser.
  """

    class UrlTypeHintValueValuesEnum(_messages.Enum):
        """Optional. Specifies the type of viewer that is used when opening the
    URL. Defaults to opening via web browser.

    Values:
      URL_TYPE_HINT_UNSPECIFIED: Unspecified
      AMP_ACTION: Url would be an amp action
      AMP_CONTENT: URL that points directly to AMP content, or to a canonical
        URL which refers to AMP content via .
    """
        URL_TYPE_HINT_UNSPECIFIED = 0
        AMP_ACTION = 1
        AMP_CONTENT = 2
    url = _messages.StringField(1)
    urlTypeHint = _messages.EnumField('UrlTypeHintValueValuesEnum', 2)