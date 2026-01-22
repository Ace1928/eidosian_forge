from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GoogleIamV3betaV3Policy(_messages.Message):
    """IAM policy - This is the Policy Service which will not launch and has
  been replaced by the principal_access_boundary_policy proto instead. Next
  ID: 11

  Messages:
    AnnotationsValue: Optional. Unstructured key-value map to store and
      retrieve arbitrary metadata. Keys must be less than or equal to 63
      characters; values must be less than or equal to 255 characters.

  Fields:
    accessBoundaryPolicy: Optional. A policy type that binds to principals and
      principal sets.
    annotations: Optional. Unstructured key-value map to store and retrieve
      arbitrary metadata. Keys must be less than or equal to 63 characters;
      values must be less than or equal to 255 characters.
    createTime: Output only. The time when the `Policy` was created.
    deleteTime: Output only. The time when the `Policy` was deleted. Empty if
      the policy is not deleted.
    description: Optional. A user-specified opaque description of the
      `Policy`. Must be less than or equal to 255 characters.
    displayName: Optional. A user-specified opaque description of the
      `Policy`. Must be less than or equal to 63 characters.
    etag: Optional. An opaque tag indicating the current version of the
      `Policy`. This is a strong etag.
    name: The resource name of the `Policy`, which must be globally unique.
      The name needs to follow formats below. This field is output_only in a
      CreatePolicyRequest. Only `global` location is supported.
      `projects/{project_id}/locations/{location}/policies/{policy_id}`
      `projects/{project_number}/locations/{location}/policies/{policy_id}`
      `folders/{numeric_id}/locations/{location}/policies/{policy_id}`
      `organizations/{numeric_id}/locations/{location}/policies/{policy_id}`
    uid: Output only. The globally unique ID of the `Policy`. Assigned when
      the `Policy` is created.
    updateTime: Output only. The time when the `Policy` was last updated.
      During creation, this field will have the create_time value.
  """

    @encoding.MapUnrecognizedFields('additionalProperties')
    class AnnotationsValue(_messages.Message):
        """Optional. Unstructured key-value map to store and retrieve arbitrary
    metadata. Keys must be less than or equal to 63 characters; values must be
    less than or equal to 255 characters.

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
    accessBoundaryPolicy = _messages.MessageField('GoogleIamV3betaAccessBoundaryPolicy', 1)
    annotations = _messages.MessageField('AnnotationsValue', 2)
    createTime = _messages.StringField(3)
    deleteTime = _messages.StringField(4)
    description = _messages.StringField(5)
    displayName = _messages.StringField(6)
    etag = _messages.StringField(7)
    name = _messages.StringField(8)
    uid = _messages.StringField(9)
    updateTime = _messages.StringField(10)