from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GoogleApiServiceusageV2alphaConsumerPolicy(_messages.Message):
    """Consumer Policy is a set of rules that define what services or service
  groups can be used for a cloud resource hierarchy.

  Messages:
    AnnotationsValue: Optional. Annotations is an unstructured key-value map
      stored with a policy that may be set by external tools to store and
      retrieve arbitrary metadata. They are not queryable and should be
      preserved when modifying objects.
      [AIP-128](https://google.aip.dev/128#annotations)

  Fields:
    annotations: Optional. Annotations is an unstructured key-value map stored
      with a policy that may be set by external tools to store and retrieve
      arbitrary metadata. They are not queryable and should be preserved when
      modifying objects. [AIP-128](https://google.aip.dev/128#annotations)
    createTime: Output only. The time the policy was created. For singleton
      policies, this is the first touch of the policy.
    enableRules: Enable rules define usable services, groups, and categories.
      There can currently be at most one `EnableRule`. This restriction will
      be lifted in later releases.
    etag: Output only. An opaque tag indicating the current version of the
      policy, used for concurrency control.
    name: Output only. The resource name of the policy. Only the `default`
      policy is supported: `projects/12345/consumerPolicies/default`,
      `folders/12345/consumerPolicies/default`,
      `organizations/12345/consumerPolicies/default`.
    updateTime: Output only. The time the policy was last updated.
  """

    @encoding.MapUnrecognizedFields('additionalProperties')
    class AnnotationsValue(_messages.Message):
        """Optional. Annotations is an unstructured key-value map stored with a
    policy that may be set by external tools to store and retrieve arbitrary
    metadata. They are not queryable and should be preserved when modifying
    objects. [AIP-128](https://google.aip.dev/128#annotations)

    Messages:
      AdditionalProperty: An additional property for a AnnotationsValue
        object.

    Fields:
      additionalProperties: Additional properties of type AnnotationsValue
    """

        class AdditionalProperty(_messages.Message):
            """An additional property for a AnnotationsValue object.

      Fields:
        key: Name of the additional property.
        value: A string attribute.
      """
            key = _messages.StringField(1)
            value = _messages.StringField(2)
        additionalProperties = _messages.MessageField('AdditionalProperty', 1, repeated=True)
    annotations = _messages.MessageField('AnnotationsValue', 1)
    createTime = _messages.StringField(2)
    enableRules = _messages.MessageField('GoogleApiServiceusageV2alphaEnableRule', 3, repeated=True)
    etag = _messages.StringField(4)
    name = _messages.StringField(5)
    updateTime = _messages.StringField(6)