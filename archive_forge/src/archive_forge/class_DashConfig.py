from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class DashConfig(_messages.Message):
    """`DASH` manifest configuration.

  Enums:
    SegmentReferenceSchemeValueValuesEnum: The segment reference scheme for a
      `DASH` manifest. The default is `SEGMENT_LIST`.

  Fields:
    segmentReferenceScheme: The segment reference scheme for a `DASH`
      manifest. The default is `SEGMENT_LIST`.
  """

    class SegmentReferenceSchemeValueValuesEnum(_messages.Enum):
        """The segment reference scheme for a `DASH` manifest. The default is
    `SEGMENT_LIST`.

    Values:
      SEGMENT_REFERENCE_SCHEME_UNSPECIFIED: The segment reference scheme is
        not specified.
      SEGMENT_LIST: Explicitly lists the URLs of media files for each segment.
        For example, if SegmentSettings.individual_segments is `true`, then
        the manifest contains fields similar to the following: ```xml ... ```
      SEGMENT_TEMPLATE_NUMBER: SegmentSettings.individual_segments must be set
        to `true` to use this segment reference scheme. Uses the DASH
        specification `` tag to determine the URLs of media files for each
        segment. For example: ```xml ... ```
    """
        SEGMENT_REFERENCE_SCHEME_UNSPECIFIED = 0
        SEGMENT_LIST = 1
        SEGMENT_TEMPLATE_NUMBER = 2
    segmentReferenceScheme = _messages.EnumField('SegmentReferenceSchemeValueValuesEnum', 1)