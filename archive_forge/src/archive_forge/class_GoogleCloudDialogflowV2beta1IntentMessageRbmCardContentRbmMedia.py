from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GoogleCloudDialogflowV2beta1IntentMessageRbmCardContentRbmMedia(_messages.Message):
    """Rich Business Messaging (RBM) Media displayed in Cards The following
  media-types are currently supported: Image Types * image/jpeg * image/jpg' *
  image/gif * image/png Video Types * video/h263 * video/m4v * video/mp4 *
  video/mpeg * video/mpeg4 * video/webm

  Enums:
    HeightValueValuesEnum: Required for cards with vertical orientation. The
      height of the media within a rich card with a vertical layout. For a
      standalone card with horizontal layout, height is not customizable, and
      this field is ignored.

  Fields:
    fileUri: Required. Publicly reachable URI of the file. The RBM platform
      determines the MIME type of the file from the content-type field in the
      HTTP headers when the platform fetches the file. The content-type field
      must be present and accurate in the HTTP response from the URL.
    height: Required for cards with vertical orientation. The height of the
      media within a rich card with a vertical layout. For a standalone card
      with horizontal layout, height is not customizable, and this field is
      ignored.
    thumbnailUri: Optional. Publicly reachable URI of the thumbnail.If you
      don't provide a thumbnail URI, the RBM platform displays a blank
      placeholder thumbnail until the user's device downloads the file.
      Depending on the user's setting, the file may not download automatically
      and may require the user to tap a download button.
  """

    class HeightValueValuesEnum(_messages.Enum):
        """Required for cards with vertical orientation. The height of the media
    within a rich card with a vertical layout. For a standalone card with
    horizontal layout, height is not customizable, and this field is ignored.

    Values:
      HEIGHT_UNSPECIFIED: Not specified.
      SHORT: 112 DP.
      MEDIUM: 168 DP.
      TALL: 264 DP. Not available for rich card carousels when the card width
        is set to small.
    """
        HEIGHT_UNSPECIFIED = 0
        SHORT = 1
        MEDIUM = 2
        TALL = 3
    fileUri = _messages.StringField(1)
    height = _messages.EnumField('HeightValueValuesEnum', 2)
    thumbnailUri = _messages.StringField(3)