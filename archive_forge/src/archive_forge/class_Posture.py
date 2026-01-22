from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class Posture(_messages.Message):
    """========================== Postures ==========================
  Definition of a Posture.

  Enums:
    CategoriesValueListEntryValuesEnum:
    StateValueValuesEnum: Required. State of Posture resource.

  Messages:
    AnnotationsValue: Optional. User annotations. These attributes can only be
      set and used by the user, and not by Google Security Postures. .

  Fields:
    annotations: Optional. User annotations. These attributes can only be set
      and used by the user, and not by Google Security Postures. .
    categories: Output only. List of categories associated with a Posture.
      Based on it's associated policies the service defines the category,
      hence it is OUTPUT_ONLY field.
    createTime: Output only. The timestamp that the posture was created.
    description: Optional. User provided description of the posture.
    etag: Optional. An opaque tag indicating the current version of the
      Posture, used for concurrency control. When the `Posture` is returned
      from either a `GetPosture` or a `ListPostures` request, this `etag`
      indicates the version of the current `Posture` to use when executing a
      read-modify-write loop. When the `Posture` is used in a `UpdatePosture`
      method, use the `etag` value that was returned from a `GetPosture`
      request as part of a read-modify-write loop for concurrency control. Not
      setting the `etag` in a `UpdatePosture` request will result in an
      unconditional write of the `Posture`.
    name: Required. Identifier. The name of this Posture resource, in the
      format of
      organizations/{org_id}/locations/{location_id}/postures/{posture}.
    policySets: Required. List of Policy sets.
    reconciling: Output only. Whether or not this Posture is in the process of
      being updated.
    revisionId: Output only. Immutable. The revision ID of the posture. The
      format is an 8-character hexadecimal string. https://google.aip.dev/162
    state: Required. State of Posture resource.
    updateTime: Output only. The timestamp that the posture was updated.
  """

    class CategoriesValueListEntryValuesEnum(_messages.Enum):
        """CategoriesValueListEntryValuesEnum enum type.

    Values:
      CATEGORY_UNSPECIFIED: Unspecified Category.
      AI: AI Category.
      AWS: Posture contains AWS policies.
      GCP: Posture contains GCP policies.
    """
        CATEGORY_UNSPECIFIED = 0
        AI = 1
        AWS = 2
        GCP = 3

    class StateValueValuesEnum(_messages.Enum):
        """Required. State of Posture resource.

    Values:
      STATE_UNSPECIFIED: Unspecified operation state.
      DEPRECATED: The Posture is marked deprecated when it is not in use by
        the user.
      DRAFT: The Posture is created successfully but is not yet ready for
        usage.
      ACTIVE: The Posture state is active. Ready for use/deployments.
    """
        STATE_UNSPECIFIED = 0
        DEPRECATED = 1
        DRAFT = 2
        ACTIVE = 3

    @encoding.MapUnrecognizedFields('additionalProperties')
    class AnnotationsValue(_messages.Message):
        """Optional. User annotations. These attributes can only be set and used
    by the user, and not by Google Security Postures. .

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
    categories = _messages.EnumField('CategoriesValueListEntryValuesEnum', 2, repeated=True)
    createTime = _messages.StringField(3)
    description = _messages.StringField(4)
    etag = _messages.StringField(5)
    name = _messages.StringField(6)
    policySets = _messages.MessageField('PolicySet', 7, repeated=True)
    reconciling = _messages.BooleanField(8)
    revisionId = _messages.StringField(9)
    state = _messages.EnumField('StateValueValuesEnum', 10)
    updateTime = _messages.StringField(11)