from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class AnnotationSet(_messages.Message):
    """An annotationSet resource is associated with an Asset and is a
  collection of timed-metadata that can be modified and searched at a high
  throughput.

  Messages:
    LabelsValue: The labels associated with this resource. Each label is a
      key-value pair.

  Fields:
    createTime: Output only. The creation time of the annotationSet.
    etag: Etag of the resource used in output and update requests.
    labels: The labels associated with this resource. Each label is a key-
      value pair.
    name: A user-specified resource name of the annotationSet `projects/{proje
      ct}/locations/{location}/assetTypes/{asset_type}/assets/{asset}/annotati
      onSets/{annotation_set}`. Here {annotation_set} is a resource id.
      Detailed rules for a resource id are: 1. 1 character minimum, 63
      characters maximum 2. only contains letters, digits, underscore and
      hyphen 3. starts with a letter if length == 1, starts with a letter or
      underscore if length > 1
    updateTime: Output only. The latest update time of the annotationSet.
  """

    @encoding.MapUnrecognizedFields('additionalProperties')
    class LabelsValue(_messages.Message):
        """The labels associated with this resource. Each label is a key-value
    pair.

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
    createTime = _messages.StringField(1)
    etag = _messages.StringField(2)
    labels = _messages.MessageField('LabelsValue', 3)
    name = _messages.StringField(4)
    updateTime = _messages.StringField(5)