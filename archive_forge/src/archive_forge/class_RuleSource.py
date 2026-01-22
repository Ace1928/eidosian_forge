from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class RuleSource(_messages.Message):
    """`RuleSource` contains source information for where a rule was found.

  Messages:
    CategorySourcesValue: Map of enabled categories as keys and the policy
      that enabled it as values. For example, the key can be
      `categories/google` and value can be
      `{"projects/123/consumerPolicies/default",
      "folders/456/consumerPolicies/default"}` where the category is enabled
      and the order of the resource list is nearest first in the hierarchy.
    GroupSourcesValue: Map of enabled groups as keys and the policy that
      enabled it as values. For example, the key can be
      `services/container.googleapis.com/groups/dependencies` and value can be
      `{"projects/123/consumerPolicies/default",
      "folders/456/consumerPolicies/default"}` where the group is enabled and
      the order of the resource list is nearest first in the hierarchy.
    ServiceSourcesValue: Map of enabled services as keys and the policy that
      enabled it as values. For example, the key can be
      `services/serviceusage.googleapis.com` and value can be
      `{"projects/123/consumerPolicies/default",
      "folders/456/consumerPolicies/default"}` where the service is enabled
      and the order of the resource list is nearest first in the hierarchy.

  Fields:
    categorySources: Map of enabled categories as keys and the policy that
      enabled it as values. For example, the key can be `categories/google`
      and value can be `{"projects/123/consumerPolicies/default",
      "folders/456/consumerPolicies/default"}` where the category is enabled
      and the order of the resource list is nearest first in the hierarchy.
    groupSources: Map of enabled groups as keys and the policy that enabled it
      as values. For example, the key can be
      `services/container.googleapis.com/groups/dependencies` and value can be
      `{"projects/123/consumerPolicies/default",
      "folders/456/consumerPolicies/default"}` where the group is enabled and
      the order of the resource list is nearest first in the hierarchy.
    serviceSources: Map of enabled services as keys and the policy that
      enabled it as values. For example, the key can be
      `services/serviceusage.googleapis.com` and value can be
      `{"projects/123/consumerPolicies/default",
      "folders/456/consumerPolicies/default"}` where the service is enabled
      and the order of the resource list is nearest first in the hierarchy.
  """

    @encoding.MapUnrecognizedFields('additionalProperties')
    class CategorySourcesValue(_messages.Message):
        """Map of enabled categories as keys and the policy that enabled it as
    values. For example, the key can be `categories/google` and value can be
    `{"projects/123/consumerPolicies/default",
    "folders/456/consumerPolicies/default"}` where the category is enabled and
    the order of the resource list is nearest first in the hierarchy.

    Messages:
      AdditionalProperty: An additional property for a CategorySourcesValue
        object.

    Fields:
      additionalProperties: Additional properties of type CategorySourcesValue
    """

        class AdditionalProperty(_messages.Message):
            """An additional property for a CategorySourcesValue object.

      Fields:
        key: Name of the additional property.
        value: A PolicyList attribute.
      """
            key = _messages.StringField(1)
            value = _messages.MessageField('PolicyList', 2)
        additionalProperties = _messages.MessageField('AdditionalProperty', 1, repeated=True)

    @encoding.MapUnrecognizedFields('additionalProperties')
    class GroupSourcesValue(_messages.Message):
        """Map of enabled groups as keys and the policy that enabled it as
    values. For example, the key can be
    `services/container.googleapis.com/groups/dependencies` and value can be
    `{"projects/123/consumerPolicies/default",
    "folders/456/consumerPolicies/default"}` where the group is enabled and
    the order of the resource list is nearest first in the hierarchy.

    Messages:
      AdditionalProperty: An additional property for a GroupSourcesValue
        object.

    Fields:
      additionalProperties: Additional properties of type GroupSourcesValue
    """

        class AdditionalProperty(_messages.Message):
            """An additional property for a GroupSourcesValue object.

      Fields:
        key: Name of the additional property.
        value: A PolicyList attribute.
      """
            key = _messages.StringField(1)
            value = _messages.MessageField('PolicyList', 2)
        additionalProperties = _messages.MessageField('AdditionalProperty', 1, repeated=True)

    @encoding.MapUnrecognizedFields('additionalProperties')
    class ServiceSourcesValue(_messages.Message):
        """Map of enabled services as keys and the policy that enabled it as
    values. For example, the key can be `services/serviceusage.googleapis.com`
    and value can be `{"projects/123/consumerPolicies/default",
    "folders/456/consumerPolicies/default"}` where the service is enabled and
    the order of the resource list is nearest first in the hierarchy.

    Messages:
      AdditionalProperty: An additional property for a ServiceSourcesValue
        object.

    Fields:
      additionalProperties: Additional properties of type ServiceSourcesValue
    """

        class AdditionalProperty(_messages.Message):
            """An additional property for a ServiceSourcesValue object.

      Fields:
        key: Name of the additional property.
        value: A PolicyList attribute.
      """
            key = _messages.StringField(1)
            value = _messages.MessageField('PolicyList', 2)
        additionalProperties = _messages.MessageField('AdditionalProperty', 1, repeated=True)
    categorySources = _messages.MessageField('CategorySourcesValue', 1)
    groupSources = _messages.MessageField('GroupSourcesValue', 2)
    serviceSources = _messages.MessageField('ServiceSourcesValue', 3)