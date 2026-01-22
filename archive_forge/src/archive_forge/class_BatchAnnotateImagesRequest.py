from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class BatchAnnotateImagesRequest(_messages.Message):
    """Multiple image annotation requests are batched into a single service
  call.

  Messages:
    LabelsValue: Optional. The labels with user-defined metadata for the
      request. Label keys and values can be no longer than 63 characters
      (Unicode codepoints), can only contain lowercase letters, numeric
      characters, underscores and dashes. International characters are
      allowed. Label values are optional. Label keys must start with a letter.

  Fields:
    labels: Optional. The labels with user-defined metadata for the request.
      Label keys and values can be no longer than 63 characters (Unicode
      codepoints), can only contain lowercase letters, numeric characters,
      underscores and dashes. International characters are allowed. Label
      values are optional. Label keys must start with a letter.
    parent: Optional. Target project and location to make a call. Format:
      `projects/{project-id}/locations/{location-id}`. If no parent is
      specified, a region will be chosen automatically. Supported location-
      ids: `us`: USA country only, `asia`: East asia areas, like Japan,
      Taiwan, `eu`: The European Union. Example:
      `projects/project-A/locations/eu`.
    requests: Required. Individual image annotation requests for this batch.
  """

    @encoding.MapUnrecognizedFields('additionalProperties')
    class LabelsValue(_messages.Message):
        """Optional. The labels with user-defined metadata for the request. Label
    keys and values can be no longer than 63 characters (Unicode codepoints),
    can only contain lowercase letters, numeric characters, underscores and
    dashes. International characters are allowed. Label values are optional.
    Label keys must start with a letter.

    Messages:
      AdditionalProperty: An additional property for a LabelsValue object.

    Fields:
      additionalProperties: Additional properties of type LabelsValue
    """

        class AdditionalProperty(_messages.Message):
            """An additional property for a LabelsValue object.

      Fields:
        key: Name of the additional property.
        value: A string attribute.
      """
            key = _messages.StringField(1)
            value = _messages.StringField(2)
        additionalProperties = _messages.MessageField('AdditionalProperty', 1, repeated=True)
    labels = _messages.MessageField('LabelsValue', 1)
    parent = _messages.StringField(2)
    requests = _messages.MessageField('AnnotateImageRequest', 3, repeated=True)