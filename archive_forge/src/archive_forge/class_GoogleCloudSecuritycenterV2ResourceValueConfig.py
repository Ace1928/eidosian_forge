from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GoogleCloudSecuritycenterV2ResourceValueConfig(_messages.Message):
    """A resource value config (RVC) is a mapping configuration of user's
  resources to resource values. Used in Attack path simulations.

  Enums:
    CloudProviderValueValuesEnum: Cloud provider this configuration applies to
    ResourceValueValueValuesEnum: Resource value level this expression
      represents Only required when there is no SDP mapping in the request

  Messages:
    ResourceLabelsSelectorValue: List of resource labels to search for,
      evaluated with AND. E.g. "resource_labels_selector": {"key": "value",
      "env": "prod"} will match resources with labels "key": "value" AND
      "env": "prod" https://cloud.google.com/resource-manager/docs/creating-
      managing-labels

  Fields:
    cloudProvider: Cloud provider this configuration applies to
    createTime: Output only. Timestamp this resource value config was created.
    description: Description of the resource value config.
    name: Name for the resource value config
    resourceLabelsSelector: List of resource labels to search for, evaluated
      with AND. E.g. "resource_labels_selector": {"key": "value", "env":
      "prod"} will match resources with labels "key": "value" AND "env":
      "prod" https://cloud.google.com/resource-manager/docs/creating-managing-
      labels
    resourceType: Apply resource_value only to resources that match
      resource_type. resource_type will be checked with "AND" of other
      resources. E.g. "storage.googleapis.com/Bucket" with resource_value
      "HIGH" will apply "HIGH" value only to "storage.googleapis.com/Bucket"
      resources.
    resourceValue: Resource value level this expression represents Only
      required when there is no SDP mapping in the request
    scope: Project or folder to scope this config to. For example,
      "project/456" would apply this config only to resources in "project/456"
      scope will be checked with "AND" of other resources.
    sensitiveDataProtectionMapping: A mapping of the sensitivity on Sensitive
      Data Protection finding to resource values. This mapping can only be
      used in combination with a resource_type that is related to BigQuery,
      e.g. "bigquery.googleapis.com/Dataset".
    tagValues: Required. Tag values combined with AND to check against. Values
      in the form "tagValues/123" E.g. [ "tagValues/123", "tagValues/456",
      "tagValues/789" ] https://cloud.google.com/resource-
      manager/docs/tags/tags-creating-and-managing
    updateTime: Output only. Timestamp this resource value config was last
      updated.
  """

    class CloudProviderValueValuesEnum(_messages.Enum):
        """Cloud provider this configuration applies to

    Values:
      CLOUD_PROVIDER_UNSPECIFIED: The cloud provider is unspecified.
      GOOGLE_CLOUD_PLATFORM: The cloud provider is Google Cloud Platform.
      AMAZON_WEB_SERVICES: The cloud provider is Amazon Web Services.
      MICROSOFT_AZURE: The cloud provider is Microsoft Azure.
    """
        CLOUD_PROVIDER_UNSPECIFIED = 0
        GOOGLE_CLOUD_PLATFORM = 1
        AMAZON_WEB_SERVICES = 2
        MICROSOFT_AZURE = 3

    class ResourceValueValueValuesEnum(_messages.Enum):
        """Resource value level this expression represents Only required when
    there is no SDP mapping in the request

    Values:
      RESOURCE_VALUE_UNSPECIFIED: Unspecific value
      HIGH: High resource value
      MEDIUM: Medium resource value
      LOW: Low resource value
      NONE: No resource value, e.g. ignore these resources
    """
        RESOURCE_VALUE_UNSPECIFIED = 0
        HIGH = 1
        MEDIUM = 2
        LOW = 3
        NONE = 4

    @encoding.MapUnrecognizedFields('additionalProperties')
    class ResourceLabelsSelectorValue(_messages.Message):
        """List of resource labels to search for, evaluated with AND. E.g.
    "resource_labels_selector": {"key": "value", "env": "prod"} will match
    resources with labels "key": "value" AND "env": "prod"
    https://cloud.google.com/resource-manager/docs/creating-managing-labels

    Messages:
      AdditionalProperty: An additional property for a
        ResourceLabelsSelectorValue object.

    Fields:
      additionalProperties: Additional properties of type
        ResourceLabelsSelectorValue
    """

        class AdditionalProperty(_messages.Message):
            """An additional property for a ResourceLabelsSelectorValue object.

      Fields:
        key: Name of the additional property.
        value: A string attribute.
      """
            key = _messages.StringField(1)
            value = _messages.StringField(2)
        additionalProperties = _messages.MessageField('AdditionalProperty', 1, repeated=True)
    cloudProvider = _messages.EnumField('CloudProviderValueValuesEnum', 1)
    createTime = _messages.StringField(2)
    description = _messages.StringField(3)
    name = _messages.StringField(4)
    resourceLabelsSelector = _messages.MessageField('ResourceLabelsSelectorValue', 5)
    resourceType = _messages.StringField(6)
    resourceValue = _messages.EnumField('ResourceValueValueValuesEnum', 7)
    scope = _messages.StringField(8)
    sensitiveDataProtectionMapping = _messages.MessageField('GoogleCloudSecuritycenterV2SensitiveDataProtectionMapping', 9)
    tagValues = _messages.StringField(10, repeated=True)
    updateTime = _messages.StringField(11)