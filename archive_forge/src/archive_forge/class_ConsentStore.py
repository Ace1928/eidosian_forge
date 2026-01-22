from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class ConsentStore(_messages.Message):
    """Represents a consent store.

  Messages:
    LabelsValue: Optional. User-supplied key-value pairs used to organize
      consent stores. Label keys must be between 1 and 63 characters long,
      have a UTF-8 encoding of maximum 128 bytes, and must conform to the
      following PCRE regular expression: \\p{Ll}\\p{Lo}{0,62}. Label values must
      be between 1 and 63 characters long, have a UTF-8 encoding of maximum
      128 bytes, and must conform to the following PCRE regular expression:
      [\\p{Ll}\\p{Lo}\\p{N}_-]{0,63}. No more than 64 labels can be associated
      with a given store. For more information:
      https://cloud.google.com/healthcare/docs/how-tos/labeling-resources

  Fields:
    labels: Optional. User-supplied key-value pairs used to organize consent
      stores. Label keys must be between 1 and 63 characters long, have a
      UTF-8 encoding of maximum 128 bytes, and must conform to the following
      PCRE regular expression: \\p{Ll}\\p{Lo}{0,62}. Label values must be
      between 1 and 63 characters long, have a UTF-8 encoding of maximum 128
      bytes, and must conform to the following PCRE regular expression:
      [\\p{Ll}\\p{Lo}\\p{N}_-]{0,63}. No more than 64 labels can be associated
      with a given store. For more information:
      https://cloud.google.com/healthcare/docs/how-tos/labeling-resources
    name: Identifier. Resource name of the consent store, of the form `project
      s/{project_id}/locations/{location_id}/datasets/{dataset_id}/consentStor
      es/{consent_store_id}`. Cannot be changed after creation.
  """

    @encoding.MapUnrecognizedFields('additionalProperties')
    class LabelsValue(_messages.Message):
        """Optional. User-supplied key-value pairs used to organize consent
    stores. Label keys must be between 1 and 63 characters long, have a UTF-8
    encoding of maximum 128 bytes, and must conform to the following PCRE
    regular expression: \\p{Ll}\\p{Lo}{0,62}. Label values must be between 1 and
    63 characters long, have a UTF-8 encoding of maximum 128 bytes, and must
    conform to the following PCRE regular expression:
    [\\p{Ll}\\p{Lo}\\p{N}_-]{0,63}. No more than 64 labels can be associated with
    a given store. For more information:
    https://cloud.google.com/healthcare/docs/how-tos/labeling-resources

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
    name = _messages.StringField(2)