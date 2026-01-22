from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
@encoding.MapUnrecognizedFields('additionalProperties')
class DocumentsMetadataFiltersValue(_messages.Message):
    """Optional. Key-value filters on the metadata of documents returned by
    article suggestion. If specified, article suggestion only returns
    suggested documents that match all filters in their Document.metadata.
    Multiple values for a metadata key should be concatenated by comma. For
    example, filters to match all documents that have 'US' or 'CA' in their
    market metadata values and 'agent' in their user metadata values will be
    ``` documents_metadata_filters { key: "market" value: "US,CA" }
    documents_metadata_filters { key: "user" value: "agent" } ```

    Messages:
      AdditionalProperty: An additional property for a
        DocumentsMetadataFiltersValue object.

    Fields:
      additionalProperties: Additional properties of type
        DocumentsMetadataFiltersValue
    """

    class AdditionalProperty(_messages.Message):
        """An additional property for a DocumentsMetadataFiltersValue object.

      Fields:
        key: Name of the additional property.
        value: A string attribute.
      """
        key = _messages.StringField(1)
        value = _messages.StringField(2)
    additionalProperties = _messages.MessageField('AdditionalProperty', 1, repeated=True)