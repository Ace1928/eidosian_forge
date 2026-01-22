from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class CloudAiLargeModelsVisionSemanticFilterResponse(_messages.Message):
    """A CloudAiLargeModelsVisionSemanticFilterResponse object.

  Fields:
    namedBoundingBoxes: Class labels of the bounding boxes that failed the
      semantic filtering. Bounding box coordinates.
    passedSemanticFilter: This response is added when semantic filter config
      is turned on in EditConfig. It reports if this image is passed semantic
      filter response. If passed_semantic_filter is false, the bounding box
      information will be populated for user to check what caused the semantic
      filter to fail.
  """
    namedBoundingBoxes = _messages.MessageField('CloudAiLargeModelsVisionNamedBoundingBox', 1, repeated=True)
    passedSemanticFilter = _messages.BooleanField(2)