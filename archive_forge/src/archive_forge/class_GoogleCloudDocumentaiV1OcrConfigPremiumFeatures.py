from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GoogleCloudDocumentaiV1OcrConfigPremiumFeatures(_messages.Message):
    """Configurations for premium OCR features.

  Fields:
    computeStyleInfo: Turn on font identification model and return font style
      information.
    enableMathOcr: Turn on the model that can extract LaTeX math formulas.
    enableSelectionMarkDetection: Turn on selection mark detector in OCR
      engine. Only available in OCR 2.0 (and later) processors.
  """
    computeStyleInfo = _messages.BooleanField(1)
    enableMathOcr = _messages.BooleanField(2)
    enableSelectionMarkDetection = _messages.BooleanField(3)