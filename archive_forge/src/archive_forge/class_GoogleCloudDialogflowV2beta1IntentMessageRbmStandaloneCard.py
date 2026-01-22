from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GoogleCloudDialogflowV2beta1IntentMessageRbmStandaloneCard(_messages.Message):
    """Standalone Rich Business Messaging (RBM) rich card. Rich cards allow you
  to respond to users with more vivid content, e.g. with media and
  suggestions. You can group multiple rich cards into one using
  RbmCarouselCard but carousel cards will give you less control over the card
  layout.

  Enums:
    CardOrientationValueValuesEnum: Required. Orientation of the card.
    ThumbnailImageAlignmentValueValuesEnum: Required if orientation is
      horizontal. Image preview alignment for standalone cards with horizontal
      layout.

  Fields:
    cardContent: Required. Card content.
    cardOrientation: Required. Orientation of the card.
    thumbnailImageAlignment: Required if orientation is horizontal. Image
      preview alignment for standalone cards with horizontal layout.
  """

    class CardOrientationValueValuesEnum(_messages.Enum):
        """Required. Orientation of the card.

    Values:
      CARD_ORIENTATION_UNSPECIFIED: Not specified.
      HORIZONTAL: Horizontal layout.
      VERTICAL: Vertical layout.
    """
        CARD_ORIENTATION_UNSPECIFIED = 0
        HORIZONTAL = 1
        VERTICAL = 2

    class ThumbnailImageAlignmentValueValuesEnum(_messages.Enum):
        """Required if orientation is horizontal. Image preview alignment for
    standalone cards with horizontal layout.

    Values:
      THUMBNAIL_IMAGE_ALIGNMENT_UNSPECIFIED: Not specified.
      LEFT: Thumbnail preview is left-aligned.
      RIGHT: Thumbnail preview is right-aligned.
    """
        THUMBNAIL_IMAGE_ALIGNMENT_UNSPECIFIED = 0
        LEFT = 1
        RIGHT = 2
    cardContent = _messages.MessageField('GoogleCloudDialogflowV2beta1IntentMessageRbmCardContent', 1)
    cardOrientation = _messages.EnumField('CardOrientationValueValuesEnum', 2)
    thumbnailImageAlignment = _messages.EnumField('ThumbnailImageAlignmentValueValuesEnum', 3)