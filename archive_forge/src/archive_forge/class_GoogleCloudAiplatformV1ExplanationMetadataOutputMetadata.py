from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GoogleCloudAiplatformV1ExplanationMetadataOutputMetadata(_messages.Message):
    """Metadata of the prediction output to be explained.

  Fields:
    displayNameMappingKey: Specify a field name in the prediction to look for
      the display name. Use this if the prediction contains the display names
      for the outputs. The display names in the prediction must have the same
      shape of the outputs, so that it can be located by
      Attribution.output_index for a specific output.
    indexDisplayNameMapping: Static mapping between the index and display
      name. Use this if the outputs are a deterministic n-dimensional array,
      e.g. a list of scores of all the classes in a pre-defined order for a
      multi-classification Model. It's not feasible if the outputs are non-
      deterministic, e.g. the Model produces top-k classes or sort the outputs
      by their values. The shape of the value must be an n-dimensional array
      of strings. The number of dimensions must match that of the outputs to
      be explained. The Attribution.output_display_name is populated by
      locating in the mapping with Attribution.output_index.
    outputTensorName: Name of the output tensor. Required and is only
      applicable to Vertex AI provided images for Tensorflow.
  """
    displayNameMappingKey = _messages.StringField(1)
    indexDisplayNameMapping = _messages.MessageField('extra_types.JsonValue', 2)
    outputTensorName = _messages.StringField(3)