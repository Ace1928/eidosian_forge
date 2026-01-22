from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
@encoding.MapUnrecognizedFields('additionalProperties')
class DataItemLabelsValue(_messages.Message):
    """Labels that will be applied to newly imported DataItems. If an
    identical DataItem as one being imported already exists in the Dataset,
    then these labels will be appended to these of the already existing one,
    and if labels with identical key is imported before, the old label value
    will be overwritten. If two DataItems are identical in the same import
    data operation, the labels will be combined and if key collision happens
    in this case, one of the values will be picked randomly. Two DataItems are
    considered identical if their content bytes are identical (e.g. image
    bytes or pdf bytes). These labels will be overridden by Annotation labels
    specified inside index file referenced by import_schema_uri, e.g. jsonl
    file.

    Messages:
      AdditionalProperty: An additional property for a DataItemLabelsValue
        object.

    Fields:
      additionalProperties: Additional properties of type DataItemLabelsValue
    """

    class AdditionalProperty(_messages.Message):
        """An additional property for a DataItemLabelsValue object.

      Fields:
        key: Name of the additional property.
        value: A string attribute.
      """
        key = _messages.StringField(1)
        value = _messages.StringField(2)
    additionalProperties = _messages.MessageField('AdditionalProperty', 1, repeated=True)