from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GoogleCloudDocumentaiV1beta2DocumentPageLayout(_messages.Message):
    """Visual element describing a layout unit on a page.

  Enums:
    OrientationValueValuesEnum: Detected orientation for the Layout.

  Fields:
    boundingPoly: The bounding polygon for the Layout.
    confidence: Confidence of the current Layout within context of the object
      this layout is for. e.g. confidence can be for a single token, a table,
      a visual element, etc. depending on context. Range `[0, 1]`.
    orientation: Detected orientation for the Layout.
    textAnchor: Text anchor indexing into the Document.text.
  """

    class OrientationValueValuesEnum(_messages.Enum):
        """Detected orientation for the Layout.

    Values:
      ORIENTATION_UNSPECIFIED: Unspecified orientation.
      PAGE_UP: Orientation is aligned with page up.
      PAGE_RIGHT: Orientation is aligned with page right. Turn the head 90
        degrees clockwise from upright to read.
      PAGE_DOWN: Orientation is aligned with page down. Turn the head 180
        degrees from upright to read.
      PAGE_LEFT: Orientation is aligned with page left. Turn the head 90
        degrees counterclockwise from upright to read.
    """
        ORIENTATION_UNSPECIFIED = 0
        PAGE_UP = 1
        PAGE_RIGHT = 2
        PAGE_DOWN = 3
        PAGE_LEFT = 4
    boundingPoly = _messages.MessageField('GoogleCloudDocumentaiV1beta2BoundingPoly', 1)
    confidence = _messages.FloatField(2, variant=_messages.Variant.FLOAT)
    orientation = _messages.EnumField('OrientationValueValuesEnum', 3)
    textAnchor = _messages.MessageField('GoogleCloudDocumentaiV1beta2DocumentTextAnchor', 4)