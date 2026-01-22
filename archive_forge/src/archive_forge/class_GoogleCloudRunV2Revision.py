from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GoogleCloudRunV2Revision(_messages.Message):
    """A Revision is an immutable snapshot of code and configuration. A
  Revision references a container image. Revisions are only created by updates
  to its parent Service.

  Enums:
    EncryptionKeyRevocationActionValueValuesEnum: The action to take if the
      encryption key is revoked.
    ExecutionEnvironmentValueValuesEnum: The execution environment being used
      to host this Revision.
    LaunchStageValueValuesEnum: The least stable launch stage needed to create
      this resource, as defined by [Google Cloud Platform Launch
      Stages](https://cloud.google.com/terms/launch-stages). Cloud Run
      supports `ALPHA`, `BETA`, and `GA`. Note that this value might not be
      what was used as input. For example, if ALPHA was provided as input in
      the parent resource, but only BETA and GA-level features are were, this
      field will be BETA.

  Messages:
    AnnotationsValue: Output only. Unstructured key value map that may be set
      by external tools to store and arbitrary metadata. They are not
      queryable and should be preserved when modifying objects.
    LabelsValue: Output only. Unstructured key value map that can be used to
      organize and categorize objects. User-provided labels are shared with
      Google's billing system, so they can be used to filter, or break down
      billing charges by team, component, environment, state, etc. For more
      information, visit https://cloud.google.com/resource-
      manager/docs/creating-managing-labels or
      https://cloud.google.com/run/docs/configuring/labels.

  Fields:
    annotations: Output only. Unstructured key value map that may be set by
      external tools to store and arbitrary metadata. They are not queryable
      and should be preserved when modifying objects.
    conditions: Output only. The Condition of this Revision, containing its
      readiness status, and detailed error information in case it did not
      reach a serving state.
    containers: Holds the single container that defines the unit of execution
      for this Revision.
    createTime: Output only. The creation time.
    deleteTime: Output only. For a deleted resource, the deletion time. It is
      only populated as a response to a Delete request.
    encryptionKey: A reference to a customer managed encryption key (CMEK) to
      use to encrypt this container image. For more information, go to
      https://cloud.google.com/run/docs/securing/using-cmek
    encryptionKeyRevocationAction: The action to take if the encryption key is
      revoked.
    encryptionKeyShutdownDuration: If encryption_key_revocation_action is
      SHUTDOWN, the duration before shutting down all instances. The minimum
      increment is 1 hour.
    etag: Output only. A system-generated fingerprint for this version of the
      resource. May be used to detect modification conflict during updates.
    executionEnvironment: The execution environment being used to host this
      Revision.
    expireTime: Output only. For a deleted resource, the time after which it
      will be permamently deleted. It is only populated as a response to a
      Delete request.
    generation: Output only. A number that monotonically increases every time
      the user modifies the desired state.
    labels: Output only. Unstructured key value map that can be used to
      organize and categorize objects. User-provided labels are shared with
      Google's billing system, so they can be used to filter, or break down
      billing charges by team, component, environment, state, etc. For more
      information, visit https://cloud.google.com/resource-
      manager/docs/creating-managing-labels or
      https://cloud.google.com/run/docs/configuring/labels.
    launchStage: The least stable launch stage needed to create this resource,
      as defined by [Google Cloud Platform Launch
      Stages](https://cloud.google.com/terms/launch-stages). Cloud Run
      supports `ALPHA`, `BETA`, and `GA`. Note that this value might not be
      what was used as input. For example, if ALPHA was provided as input in
      the parent resource, but only BETA and GA-level features are were, this
      field will be BETA.
    logUri: Output only. The Google Console URI to obtain logs for the
      Revision.
    maxInstanceRequestConcurrency: Sets the maximum number of requests that
      each serving instance can receive.
    name: Output only. The unique name of this Revision.
    observedGeneration: Output only. The generation of this Revision currently
      serving traffic. See comments in `reconciling` for additional
      information on reconciliation process in Cloud Run.
    reconciling: Output only. Indicates whether the resource's reconciliation
      is still in progress. See comments in `Service.reconciling` for
      additional information on reconciliation process in Cloud Run.
    satisfiesPzs: Output only. Reserved for future use.
    scaling: Scaling settings for this revision.
    scalingStatus: Output only. The current effective scaling settings for the
      revision.
    service: Output only. The name of the parent service.
    serviceAccount: Email address of the IAM service account associated with
      the revision of the service. The service account represents the identity
      of the running revision, and determines what permissions the revision
      has.
    sessionAffinity: Enable session affinity.
    timeout: Max allowed time for an instance to respond to a request.
    uid: Output only. Server assigned unique identifier for the Revision. The
      value is a UUID4 string and guaranteed to remain unchanged until the
      resource is deleted.
    updateTime: Output only. The last-modified time.
    volumes: A list of Volumes to make available to containers.
    vpcAccess: VPC Access configuration for this Revision. For more
      information, visit
      https://cloud.google.com/run/docs/configuring/connecting-vpc.
  """

    class EncryptionKeyRevocationActionValueValuesEnum(_messages.Enum):
        """The action to take if the encryption key is revoked.

    Values:
      ENCRYPTION_KEY_REVOCATION_ACTION_UNSPECIFIED: Unspecified
      PREVENT_NEW: Prevents the creation of new instances.
      SHUTDOWN: Shuts down existing instances, and prevents creation of new
        ones.
    """
        ENCRYPTION_KEY_REVOCATION_ACTION_UNSPECIFIED = 0
        PREVENT_NEW = 1
        SHUTDOWN = 2

    class ExecutionEnvironmentValueValuesEnum(_messages.Enum):
        """The execution environment being used to host this Revision.

    Values:
      EXECUTION_ENVIRONMENT_UNSPECIFIED: Unspecified
      EXECUTION_ENVIRONMENT_GEN1: Uses the First Generation environment.
      EXECUTION_ENVIRONMENT_GEN2: Uses Second Generation environment.
    """
        EXECUTION_ENVIRONMENT_UNSPECIFIED = 0
        EXECUTION_ENVIRONMENT_GEN1 = 1
        EXECUTION_ENVIRONMENT_GEN2 = 2

    class LaunchStageValueValuesEnum(_messages.Enum):
        """The least stable launch stage needed to create this resource, as
    defined by [Google Cloud Platform Launch
    Stages](https://cloud.google.com/terms/launch-stages). Cloud Run supports
    `ALPHA`, `BETA`, and `GA`. Note that this value might not be what was used
    as input. For example, if ALPHA was provided as input in the parent
    resource, but only BETA and GA-level features are were, this field will be
    BETA.

    Values:
      LAUNCH_STAGE_UNSPECIFIED: Do not use this default value.
      UNIMPLEMENTED: The feature is not yet implemented. Users can not use it.
      PRELAUNCH: Prelaunch features are hidden from users and are only visible
        internally.
      EARLY_ACCESS: Early Access features are limited to a closed group of
        testers. To use these features, you must sign up in advance and sign a
        Trusted Tester agreement (which includes confidentiality provisions).
        These features may be unstable, changed in backward-incompatible ways,
        and are not guaranteed to be released.
      ALPHA: Alpha is a limited availability test for releases before they are
        cleared for widespread use. By Alpha, all significant design issues
        are resolved and we are in the process of verifying functionality.
        Alpha customers need to apply for access, agree to applicable terms,
        and have their projects allowlisted. Alpha releases don't have to be
        feature complete, no SLAs are provided, and there are no technical
        support obligations, but they will be far enough along that customers
        can actually use them in test environments or for limited-use tests --
        just like they would in normal production cases.
      BETA: Beta is the point at which we are ready to open a release for any
        customer to use. There are no SLA or technical support obligations in
        a Beta release. Products will be complete from a feature perspective,
        but may have some open outstanding issues. Beta releases are suitable
        for limited production use cases.
      GA: GA features are open to all developers and are considered stable and
        fully qualified for production use.
      DEPRECATED: Deprecated features are scheduled to be shut down and
        removed. For more information, see the "Deprecation Policy" section of
        our [Terms of Service](https://cloud.google.com/terms/) and the
        [Google Cloud Platform Subject to the Deprecation
        Policy](https://cloud.google.com/terms/deprecation) documentation.
    """
        LAUNCH_STAGE_UNSPECIFIED = 0
        UNIMPLEMENTED = 1
        PRELAUNCH = 2
        EARLY_ACCESS = 3
        ALPHA = 4
        BETA = 5
        GA = 6
        DEPRECATED = 7

    @encoding.MapUnrecognizedFields('additionalProperties')
    class AnnotationsValue(_messages.Message):
        """Output only. Unstructured key value map that may be set by external
    tools to store and arbitrary metadata. They are not queryable and should
    be preserved when modifying objects.

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

    @encoding.MapUnrecognizedFields('additionalProperties')
    class LabelsValue(_messages.Message):
        """Output only. Unstructured key value map that can be used to organize
    and categorize objects. User-provided labels are shared with Google's
    billing system, so they can be used to filter, or break down billing
    charges by team, component, environment, state, etc. For more information,
    visit https://cloud.google.com/resource-manager/docs/creating-managing-
    labels or https://cloud.google.com/run/docs/configuring/labels.

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
    annotations = _messages.MessageField('AnnotationsValue', 1)
    conditions = _messages.MessageField('GoogleCloudRunV2Condition', 2, repeated=True)
    containers = _messages.MessageField('GoogleCloudRunV2Container', 3, repeated=True)
    createTime = _messages.StringField(4)
    deleteTime = _messages.StringField(5)
    encryptionKey = _messages.StringField(6)
    encryptionKeyRevocationAction = _messages.EnumField('EncryptionKeyRevocationActionValueValuesEnum', 7)
    encryptionKeyShutdownDuration = _messages.StringField(8)
    etag = _messages.StringField(9)
    executionEnvironment = _messages.EnumField('ExecutionEnvironmentValueValuesEnum', 10)
    expireTime = _messages.StringField(11)
    generation = _messages.IntegerField(12)
    labels = _messages.MessageField('LabelsValue', 13)
    launchStage = _messages.EnumField('LaunchStageValueValuesEnum', 14)
    logUri = _messages.StringField(15)
    maxInstanceRequestConcurrency = _messages.IntegerField(16, variant=_messages.Variant.INT32)
    name = _messages.StringField(17)
    observedGeneration = _messages.IntegerField(18)
    reconciling = _messages.BooleanField(19)
    satisfiesPzs = _messages.BooleanField(20)
    scaling = _messages.MessageField('GoogleCloudRunV2RevisionScaling', 21)
    scalingStatus = _messages.MessageField('GoogleCloudRunV2RevisionScalingStatus', 22)
    service = _messages.StringField(23)
    serviceAccount = _messages.StringField(24)
    sessionAffinity = _messages.BooleanField(25)
    timeout = _messages.StringField(26)
    uid = _messages.StringField(27)
    updateTime = _messages.StringField(28)
    volumes = _messages.MessageField('GoogleCloudRunV2Volume', 29, repeated=True)
    vpcAccess = _messages.MessageField('GoogleCloudRunV2VpcAccess', 30)