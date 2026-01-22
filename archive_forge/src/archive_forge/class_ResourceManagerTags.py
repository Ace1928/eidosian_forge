from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class ResourceManagerTags(_messages.Message):
    """A map of resource manager tag keys and values to be attached to the
  nodes for managing Compute Engine firewalls using Network Firewall Policies.
  Tags must be according to specifications in
  https://cloud.google.com/vpc/docs/tags-firewalls-overview#specifications. A
  maximum of 5 tag key-value pairs can be specified. Existing tags will be
  replaced with new values.

  Messages:
    TagsValue: TagKeyValue must be in one of the following formats
      ([KEY]=[VALUE]) 1. `tagKeys/{tag_key_id}=tagValues/{tag_value_id}` 2.
      `{org_id}/{tag_key_name}={tag_value_name}` 3.
      `{project_id}/{tag_key_name}={tag_value_name}`

  Fields:
    tags: TagKeyValue must be in one of the following formats ([KEY]=[VALUE])
      1. `tagKeys/{tag_key_id}=tagValues/{tag_value_id}` 2.
      `{org_id}/{tag_key_name}={tag_value_name}` 3.
      `{project_id}/{tag_key_name}={tag_value_name}`
  """

    @encoding.MapUnrecognizedFields('additionalProperties')
    class TagsValue(_messages.Message):
        """TagKeyValue must be in one of the following formats ([KEY]=[VALUE]) 1.
    `tagKeys/{tag_key_id}=tagValues/{tag_value_id}` 2.
    `{org_id}/{tag_key_name}={tag_value_name}` 3.
    `{project_id}/{tag_key_name}={tag_value_name}`

    Messages:
      AdditionalProperty: An additional property for a TagsValue object.

    Fields:
      additionalProperties: Additional properties of type TagsValue
    """

        class AdditionalProperty(_messages.Message):
            """An additional property for a TagsValue object.

      Fields:
        key: Name of the additional property.
        value: A string attribute.
      """
            key = _messages.StringField(1)
            value = _messages.StringField(2)
        additionalProperties = _messages.MessageField('AdditionalProperty', 1, repeated=True)
    tags = _messages.MessageField('TagsValue', 1)