from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
@encoding.MapUnrecognizedFields('additionalProperties')
class FileHashesValue(_messages.Message):
    """Output only. Hash(es) of the build source, which can be used to verify
    that the original source integrity was maintained in the build. Note that
    `FileHashes` will only be populated if `BuildOptions` has requested a
    `SourceProvenanceHash`. The keys to this map are file paths used as build
    source and the values contain the hash values for those files. If the
    build source came in a single package such as a gzipped tarfile
    (`.tar.gz`), the `FileHash` will be for the single path to that file.

    Messages:
      AdditionalProperty: An additional property for a FileHashesValue object.

    Fields:
      additionalProperties: Additional properties of type FileHashesValue
    """

    class AdditionalProperty(_messages.Message):
        """An additional property for a FileHashesValue object.

      Fields:
        key: Name of the additional property.
        value: A GoogleDevtoolsCloudbuildV1FileHashes attribute.
      """
        key = _messages.StringField(1)
        value = _messages.MessageField('GoogleDevtoolsCloudbuildV1FileHashes', 2)
    additionalProperties = _messages.MessageField('AdditionalProperty', 1, repeated=True)