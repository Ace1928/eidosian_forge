from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GoogleCloudApigeeV1ArchiveDeployment(_messages.Message):
    """Archive Deployment information.

  Messages:
    LabelsValue: User-supplied key-value pairs used to organize
      ArchiveDeployments. Label keys must be between 1 and 63 characters long,
      have a UTF-8 encoding of maximum 128 bytes, and must conform to the
      following PCRE regular expression: \\p{Ll}\\p{Lo}{0,62} Label values must
      be between 1 and 63 characters long, have a UTF-8 encoding of maximum
      128 bytes, and must conform to the following PCRE regular expression:
      [\\p{Ll}\\p{Lo}\\p{N}_-]{0,63} No more than 64 labels can be associated
      with a given store.

  Fields:
    createdAt: Output only. The time at which the Archive Deployment was
      created in milliseconds since the epoch.
    gcsUri: Input only. The Google Cloud Storage signed URL returned from
      GenerateUploadUrl and used to upload the Archive zip file.
    labels: User-supplied key-value pairs used to organize ArchiveDeployments.
      Label keys must be between 1 and 63 characters long, have a UTF-8
      encoding of maximum 128 bytes, and must conform to the following PCRE
      regular expression: \\p{Ll}\\p{Lo}{0,62} Label values must be between 1
      and 63 characters long, have a UTF-8 encoding of maximum 128 bytes, and
      must conform to the following PCRE regular expression:
      [\\p{Ll}\\p{Lo}\\p{N}_-]{0,63} No more than 64 labels can be associated
      with a given store.
    name: Name of the Archive Deployment in the following format:
      `organizations/{org}/environments/{env}/archiveDeployments/{id}`.
    operation: Output only. A reference to the LRO that created this Archive
      Deployment in the following format:
      `organizations/{org}/operations/{id}`
    updatedAt: Output only. The time at which the Archive Deployment was
      updated in milliseconds since the epoch.
  """

    @encoding.MapUnrecognizedFields('additionalProperties')
    class LabelsValue(_messages.Message):
        """User-supplied key-value pairs used to organize ArchiveDeployments.
    Label keys must be between 1 and 63 characters long, have a UTF-8 encoding
    of maximum 128 bytes, and must conform to the following PCRE regular
    expression: \\p{Ll}\\p{Lo}{0,62} Label values must be between 1 and 63
    characters long, have a UTF-8 encoding of maximum 128 bytes, and must
    conform to the following PCRE regular expression:
    [\\p{Ll}\\p{Lo}\\p{N}_-]{0,63} No more than 64 labels can be associated with
    a given store.

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
    createdAt = _messages.IntegerField(1)
    gcsUri = _messages.StringField(2)
    labels = _messages.MessageField('LabelsValue', 3)
    name = _messages.StringField(4)
    operation = _messages.StringField(5)
    updatedAt = _messages.IntegerField(6)