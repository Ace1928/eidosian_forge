from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class LayoutTypeValueValuesEnum(_messages.Enum):
    """Optional. The type of the layout element that is being referenced if
    any.

    Values:
      LAYOUT_TYPE_UNSPECIFIED: Layout Unspecified.
      BLOCK: References a Page.blocks element.
      PARAGRAPH: References a Page.paragraphs element.
      LINE: References a Page.lines element.
      TOKEN: References a Page.tokens element.
      VISUAL_ELEMENT: References a Page.visual_elements element.
      TABLE: Refrrences a Page.tables element.
      FORM_FIELD: References a Page.form_fields element.
    """
    LAYOUT_TYPE_UNSPECIFIED = 0
    BLOCK = 1
    PARAGRAPH = 2
    LINE = 3
    TOKEN = 4
    VISUAL_ELEMENT = 5
    TABLE = 6
    FORM_FIELD = 7