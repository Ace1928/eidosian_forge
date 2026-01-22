from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class PostureDeployment(_messages.Message):
    """========================== PostureDeployments ==========================
  Message describing PostureDeployment resource.

  Enums:
    CategoriesValueListEntryValuesEnum:
    StateValueValuesEnum: Output only. State of PostureDeployment resource.

  Messages:
    AnnotationsValue: Optional. User annotations. These attributes can only be
      set and used by the user, and not by Google Security Postures. .

  Fields:
    annotations: Optional. User annotations. These attributes can only be set
      and used by the user, and not by Google Security Postures. .
    categories: Output only. Categories are a function of policies present in
      a Posture Version associated with this PostureDeployment.
    createTime: Output only. The timestamp that the PostureDeployment was
      created.
    description: Optional. User provided description of the PostureDeployment.
    desiredPostureId: Output only. This is a output only optional field which
      will be filled in case where PostureDeployment state is UPDATE_FAILED or
      CREATE_FAILED or DELETE_FAILED. It denotes the desired Posture.
    desiredPostureRevisionId: Output only. Output only optional field which
      provides revision_id of the desired_posture_id.
    etag: Optional. An opaque tag indicating the current version of the
      PostureDeployment, used for concurrency control. When the
      `PostureDeployment` is returned from either a `GetPostureDeployment` or
      a `ListPostureDeployments` request, this `etag` indicates the version of
      the current `PostureDeployment` to use when executing a read-modify-
      write loop. When the `PostureDeployment` is used in a
      `UpdatePostureDeployment` method, use the `etag` value that was returned
      from a `GetPostureDeployment` request as part of a read-modify-write
      loop for concurrency control. Not setting the `etag` in a
      `UpdatePostureDeployment` request will result in an unconditional write
      of the `PostureDeployment`.
    failureMessage: Output only. This is a output only optional field which
      will be filled in case where PostureDeployment enters a failure state
      like UPDATE_FAILED or CREATE_FAILED or DELETE_FAILED.
    name: Required. The name of this PostureDeployment resource, in the format
      of organizations/{organization}/locations/{location_id}/postureDeploymen
      ts/{postureDeployment}.
    postureId: Required. Posture that needs to be deployed. Format:
      organizations/{org_id}/locations/{location_id}/postures/ Example:
      organizations/99/locations/global/postures/les-miserables.
    postureRevisionId: Required. Revision_id of the Posture that is to be
      deployed.
    reconciling: Output only. Whether or not this Posture is in the process of
      being updated.
    state: Output only. State of PostureDeployment resource.
    targetResource: Required. Target resource where the Posture will be
      deployed. Currently supported resources are of types:
      projects/projectNumber, folders/folderNumber,
      organizations/organizationNumber.
    updateTime: Output only. The timestamp that the PostureDeployment was
      updated.
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
        """Output only. State of PostureDeployment resource.

    Values:
      STATE_UNSPECIFIED: Unspecified operation state.
      CREATING: The PostureDeployment is being created.
      DELETING: The PostureDeployment is being deleted.
      UPDATING: The PostureDeployment state is being updated.
      ACTIVE: The PostureDeployment state is active and in use.
      CREATE_FAILED: The PostureDeployment creation failed.
      UPDATE_FAILED: The PostureDeployment update failed.
      DELETE_FAILED: The PostureDeployment deletion failed.
    """
        STATE_UNSPECIFIED = 0
        CREATING = 1
        DELETING = 2
        UPDATING = 3
        ACTIVE = 4
        CREATE_FAILED = 5
        UPDATE_FAILED = 6
        DELETE_FAILED = 7

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
    desiredPostureId = _messages.StringField(5)
    desiredPostureRevisionId = _messages.StringField(6)
    etag = _messages.StringField(7)
    failureMessage = _messages.StringField(8)
    name = _messages.StringField(9)
    postureId = _messages.StringField(10)
    postureRevisionId = _messages.StringField(11)
    reconciling = _messages.BooleanField(12)
    state = _messages.EnumField('StateValueValuesEnum', 13)
    targetResource = _messages.StringField(14)
    updateTime = _messages.StringField(15)