from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GoogleCloudAiplatformV1beta1ExplanationMetadataInputMetadataVisualization(_messages.Message):
    """Visualization configurations for image explanation.

  Enums:
    ColorMapValueValuesEnum: The color scheme used for the highlighted areas.
      Defaults to PINK_GREEN for Integrated Gradients attribution, which shows
      positive attributions in green and negative in pink. Defaults to VIRIDIS
      for XRAI attribution, which highlights the most influential regions in
      yellow and the least influential in blue.
    OverlayTypeValueValuesEnum: How the original image is displayed in the
      visualization. Adjusting the overlay can help increase visual clarity if
      the original image makes it difficult to view the visualization.
      Defaults to NONE.
    PolarityValueValuesEnum: Whether to only highlight pixels with positive
      contributions, negative or both. Defaults to POSITIVE.
    TypeValueValuesEnum: Type of the image visualization. Only applicable to
      Integrated Gradients attribution. OUTLINES shows regions of attribution,
      while PIXELS shows per-pixel attribution. Defaults to OUTLINES.

  Fields:
    clipPercentLowerbound: Excludes attributions below the specified
      percentile, from the highlighted areas. Defaults to 62.
    clipPercentUpperbound: Excludes attributions above the specified
      percentile from the highlighted areas. Using the clip_percent_upperbound
      and clip_percent_lowerbound together can be useful for filtering out
      noise and making it easier to see areas of strong attribution. Defaults
      to 99.9.
    colorMap: The color scheme used for the highlighted areas. Defaults to
      PINK_GREEN for Integrated Gradients attribution, which shows positive
      attributions in green and negative in pink. Defaults to VIRIDIS for XRAI
      attribution, which highlights the most influential regions in yellow and
      the least influential in blue.
    overlayType: How the original image is displayed in the visualization.
      Adjusting the overlay can help increase visual clarity if the original
      image makes it difficult to view the visualization. Defaults to NONE.
    polarity: Whether to only highlight pixels with positive contributions,
      negative or both. Defaults to POSITIVE.
    type: Type of the image visualization. Only applicable to Integrated
      Gradients attribution. OUTLINES shows regions of attribution, while
      PIXELS shows per-pixel attribution. Defaults to OUTLINES.
  """

    class ColorMapValueValuesEnum(_messages.Enum):
        """The color scheme used for the highlighted areas. Defaults to
    PINK_GREEN for Integrated Gradients attribution, which shows positive
    attributions in green and negative in pink. Defaults to VIRIDIS for XRAI
    attribution, which highlights the most influential regions in yellow and
    the least influential in blue.

    Values:
      COLOR_MAP_UNSPECIFIED: Should not be used.
      PINK_GREEN: Positive: green. Negative: pink.
      VIRIDIS: Viridis color map: A perceptually uniform color mapping which
        is easier to see by those with colorblindness and progresses from
        yellow to green to blue. Positive: yellow. Negative: blue.
      RED: Positive: red. Negative: red.
      GREEN: Positive: green. Negative: green.
      RED_GREEN: Positive: green. Negative: red.
      PINK_WHITE_GREEN: PiYG palette.
    """
        COLOR_MAP_UNSPECIFIED = 0
        PINK_GREEN = 1
        VIRIDIS = 2
        RED = 3
        GREEN = 4
        RED_GREEN = 5
        PINK_WHITE_GREEN = 6

    class OverlayTypeValueValuesEnum(_messages.Enum):
        """How the original image is displayed in the visualization. Adjusting
    the overlay can help increase visual clarity if the original image makes
    it difficult to view the visualization. Defaults to NONE.

    Values:
      OVERLAY_TYPE_UNSPECIFIED: Default value. This is the same as NONE.
      NONE: No overlay.
      ORIGINAL: The attributions are shown on top of the original image.
      GRAYSCALE: The attributions are shown on top of grayscaled version of
        the original image.
      MASK_BLACK: The attributions are used as a mask to reveal predictive
        parts of the image and hide the un-predictive parts.
    """
        OVERLAY_TYPE_UNSPECIFIED = 0
        NONE = 1
        ORIGINAL = 2
        GRAYSCALE = 3
        MASK_BLACK = 4

    class PolarityValueValuesEnum(_messages.Enum):
        """Whether to only highlight pixels with positive contributions, negative
    or both. Defaults to POSITIVE.

    Values:
      POLARITY_UNSPECIFIED: Default value. This is the same as POSITIVE.
      POSITIVE: Highlights the pixels/outlines that were most influential to
        the model's prediction.
      NEGATIVE: Setting polarity to negative highlights areas that does not
        lead to the models's current prediction.
      BOTH: Shows both positive and negative attributions.
    """
        POLARITY_UNSPECIFIED = 0
        POSITIVE = 1
        NEGATIVE = 2
        BOTH = 3

    class TypeValueValuesEnum(_messages.Enum):
        """Type of the image visualization. Only applicable to Integrated
    Gradients attribution. OUTLINES shows regions of attribution, while PIXELS
    shows per-pixel attribution. Defaults to OUTLINES.

    Values:
      TYPE_UNSPECIFIED: Should not be used.
      PIXELS: Shows which pixel contributed to the image prediction.
      OUTLINES: Shows which region contributed to the image prediction by
        outlining the region.
    """
        TYPE_UNSPECIFIED = 0
        PIXELS = 1
        OUTLINES = 2
    clipPercentLowerbound = _messages.FloatField(1, variant=_messages.Variant.FLOAT)
    clipPercentUpperbound = _messages.FloatField(2, variant=_messages.Variant.FLOAT)
    colorMap = _messages.EnumField('ColorMapValueValuesEnum', 3)
    overlayType = _messages.EnumField('OverlayTypeValueValuesEnum', 4)
    polarity = _messages.EnumField('PolarityValueValuesEnum', 5)
    type = _messages.EnumField('TypeValueValuesEnum', 6)