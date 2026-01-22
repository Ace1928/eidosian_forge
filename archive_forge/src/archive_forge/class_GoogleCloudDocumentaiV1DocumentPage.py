from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GoogleCloudDocumentaiV1DocumentPage(_messages.Message):
    """A page in a Document.

  Fields:
    blocks: A list of visually detected text blocks on the page. A block has a
      set of lines (collected into paragraphs) that have a common line-spacing
      and orientation.
    detectedBarcodes: A list of detected barcodes.
    detectedLanguages: A list of detected languages together with confidence.
    dimension: Physical dimension of the page.
    formFields: A list of visually detected form fields on the page.
    image: Rendered image for this page. This image is preprocessed to remove
      any skew, rotation, and distortions such that the annotation bounding
      boxes can be upright and axis-aligned.
    imageQualityScores: Image quality scores.
    layout: Layout for the page.
    lines: A list of visually detected text lines on the page. A collection of
      tokens that a human would perceive as a line.
    pageNumber: 1-based index for current Page in a parent Document. Useful
      when a page is taken out of a Document for individual processing.
    paragraphs: A list of visually detected text paragraphs on the page. A
      collection of lines that a human would perceive as a paragraph.
    provenance: The history of this page.
    symbols: A list of visually detected symbols on the page.
    tables: A list of visually detected tables on the page.
    tokens: A list of visually detected tokens on the page.
    transforms: Transformation matrices that were applied to the original
      document image to produce Page.image.
    visualElements: A list of detected non-text visual elements e.g. checkbox,
      signature etc. on the page.
  """
    blocks = _messages.MessageField('GoogleCloudDocumentaiV1DocumentPageBlock', 1, repeated=True)
    detectedBarcodes = _messages.MessageField('GoogleCloudDocumentaiV1DocumentPageDetectedBarcode', 2, repeated=True)
    detectedLanguages = _messages.MessageField('GoogleCloudDocumentaiV1DocumentPageDetectedLanguage', 3, repeated=True)
    dimension = _messages.MessageField('GoogleCloudDocumentaiV1DocumentPageDimension', 4)
    formFields = _messages.MessageField('GoogleCloudDocumentaiV1DocumentPageFormField', 5, repeated=True)
    image = _messages.MessageField('GoogleCloudDocumentaiV1DocumentPageImage', 6)
    imageQualityScores = _messages.MessageField('GoogleCloudDocumentaiV1DocumentPageImageQualityScores', 7)
    layout = _messages.MessageField('GoogleCloudDocumentaiV1DocumentPageLayout', 8)
    lines = _messages.MessageField('GoogleCloudDocumentaiV1DocumentPageLine', 9, repeated=True)
    pageNumber = _messages.IntegerField(10, variant=_messages.Variant.INT32)
    paragraphs = _messages.MessageField('GoogleCloudDocumentaiV1DocumentPageParagraph', 11, repeated=True)
    provenance = _messages.MessageField('GoogleCloudDocumentaiV1DocumentProvenance', 12)
    symbols = _messages.MessageField('GoogleCloudDocumentaiV1DocumentPageSymbol', 13, repeated=True)
    tables = _messages.MessageField('GoogleCloudDocumentaiV1DocumentPageTable', 14, repeated=True)
    tokens = _messages.MessageField('GoogleCloudDocumentaiV1DocumentPageToken', 15, repeated=True)
    transforms = _messages.MessageField('GoogleCloudDocumentaiV1DocumentPageMatrix', 16, repeated=True)
    visualElements = _messages.MessageField('GoogleCloudDocumentaiV1DocumentPageVisualElement', 17, repeated=True)