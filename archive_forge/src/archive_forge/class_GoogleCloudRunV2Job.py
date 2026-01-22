from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GoogleCloudRunV2Job(_messages.Message):
    """Job represents the configuration of a single job, which references a
  container image that is run to completion.

  Enums:
    LaunchStageValueValuesEnum: The launch stage as defined by [Google Cloud
      Platform Launch Stages](https://cloud.google.com/terms/launch-stages).
      Cloud Run supports `ALPHA`, `BETA`, and `GA`. If no value is specified,
      GA is assumed. Set the launch stage to a preview stage on input to allow
      use of preview features in that stage. On read (or output), describes
      whether the resource uses preview features. For example, if ALPHA is
      provided as input, but only BETA and GA-level features are used, this
      field will be BETA on output.

  Messages:
    AnnotationsValue: Unstructured key value map that may be set by external
      tools to store and arbitrary metadata. They are not queryable and should
      be preserved when modifying objects. Cloud Run API v2 does not support
      annotations with `run.googleapis.com`, `cloud.googleapis.com`,
      `serving.knative.dev`, or `autoscaling.knative.dev` namespaces, and they
      will be rejected on new resources. All system annotations in v1 now have
      a corresponding field in v2 Job. This field follows Kubernetes
      annotations' namespacing, limits, and rules.
    LabelsValue: Unstructured key value map that can be used to organize and
      categorize objects. User-provided labels are shared with Google's
      billing system, so they can be used to filter, or break down billing
      charges by team, component, environment, state, etc. For more
      information, visit https://cloud.google.com/resource-
      manager/docs/creating-managing-labels or
      https://cloud.google.com/run/docs/configuring/labels. Cloud Run API v2
      does not support labels with `run.googleapis.com`,
      `cloud.googleapis.com`, `serving.knative.dev`, or
      `autoscaling.knative.dev` namespaces, and they will be rejected. All
      system labels in v1 now have a corresponding field in v2 Job.

  Fields:
    annotations: Unstructured key value map that may be set by external tools
      to store and arbitrary metadata. They are not queryable and should be
      preserved when modifying objects. Cloud Run API v2 does not support
      annotations with `run.googleapis.com`, `cloud.googleapis.com`,
      `serving.knative.dev`, or `autoscaling.knative.dev` namespaces, and they
      will be rejected on new resources. All system annotations in v1 now have
      a corresponding field in v2 Job. This field follows Kubernetes
      annotations' namespacing, limits, and rules.
    binaryAuthorization: Settings for the Binary Authorization feature.
    client: Arbitrary identifier for the API client.
    clientVersion: Arbitrary version identifier for the API client.
    conditions: Output only. The Conditions of all other associated sub-
      resources. They contain additional diagnostics information in case the
      Job does not reach its desired state. See comments in `reconciling` for
      additional information on reconciliation process in Cloud Run.
    createTime: Output only. The creation time.
    creator: Output only. Email address of the authenticated creator.
    deleteTime: Output only. The deletion time.
    etag: Output only. A system-generated fingerprint for this version of the
      resource. May be used to detect modification conflict during updates.
    executionCount: Output only. Number of executions created for this job.
    expireTime: Output only. For a deleted resource, the time after which it
      will be permamently deleted.
    generation: Output only. A number that monotonically increases every time
      the user modifies the desired state.
    labels: Unstructured key value map that can be used to organize and
      categorize objects. User-provided labels are shared with Google's
      billing system, so they can be used to filter, or break down billing
      charges by team, component, environment, state, etc. For more
      information, visit https://cloud.google.com/resource-
      manager/docs/creating-managing-labels or
      https://cloud.google.com/run/docs/configuring/labels. Cloud Run API v2
      does not support labels with `run.googleapis.com`,
      `cloud.googleapis.com`, `serving.knative.dev`, or
      `autoscaling.knative.dev` namespaces, and they will be rejected. All
      system labels in v1 now have a corresponding field in v2 Job.
    lastModifier: Output only. Email address of the last authenticated
      modifier.
    latestCreatedExecution: Output only. Name of the last created execution.
    launchStage: The launch stage as defined by [Google Cloud Platform Launch
      Stages](https://cloud.google.com/terms/launch-stages). Cloud Run
      supports `ALPHA`, `BETA`, and `GA`. If no value is specified, GA is
      assumed. Set the launch stage to a preview stage on input to allow use
      of preview features in that stage. On read (or output), describes
      whether the resource uses preview features. For example, if ALPHA is
      provided as input, but only BETA and GA-level features are used, this
      field will be BETA on output.
    name: The fully qualified name of this Job. Format:
      projects/{project}/locations/{location}/jobs/{job}
    observedGeneration: Output only. The generation of this Job. See comments
      in `reconciling` for additional information on reconciliation process in
      Cloud Run.
    reconciling: Output only. Returns true if the Job is currently being acted
      upon by the system to bring it into the desired state. When a new Job is
      created, or an existing one is updated, Cloud Run will asynchronously
      perform all necessary steps to bring the Job to the desired state. This
      process is called reconciliation. While reconciliation is in process,
      `observed_generation` and `latest_succeeded_execution`, will have
      transient values that might mismatch the intended state: Once
      reconciliation is over (and this field is false), there are two possible
      outcomes: reconciliation succeeded and the state matches the Job, or
      there was an error, and reconciliation failed. This state can be found
      in `terminal_condition.state`. If reconciliation succeeded, the
      following fields will match: `observed_generation` and `generation`,
      `latest_succeeded_execution` and `latest_created_execution`. If
      reconciliation failed, `observed_generation` and
      `latest_succeeded_execution` will have the state of the last succeeded
      execution or empty for newly created Job. Additional information on the
      failure can be found in `terminal_condition` and `conditions`.
    satisfiesPzs: Output only. Reserved for future use.
    template: Required. The template used to create executions for this Job.
    terminalCondition: Output only. The Condition of this Job, containing its
      readiness status, and detailed error information in case it did not
      reach the desired state.
    uid: Output only. Server assigned unique identifier for the Execution. The
      value is a UUID4 string and guaranteed to remain unchanged until the
      resource is deleted.
    updateTime: Output only. The last-modified time.
  """

    class LaunchStageValueValuesEnum(_messages.Enum):
        """The launch stage as defined by [Google Cloud Platform Launch
    Stages](https://cloud.google.com/terms/launch-stages). Cloud Run supports
    `ALPHA`, `BETA`, and `GA`. If no value is specified, GA is assumed. Set
    the launch stage to a preview stage on input to allow use of preview
    features in that stage. On read (or output), describes whether the
    resource uses preview features. For example, if ALPHA is provided as
    input, but only BETA and GA-level features are used, this field will be
    BETA on output.

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
        """Unstructured key value map that may be set by external tools to store
    and arbitrary metadata. They are not queryable and should be preserved
    when modifying objects. Cloud Run API v2 does not support annotations with
    `run.googleapis.com`, `cloud.googleapis.com`, `serving.knative.dev`, or
    `autoscaling.knative.dev` namespaces, and they will be rejected on new
    resources. All system annotations in v1 now have a corresponding field in
    v2 Job. This field follows Kubernetes annotations' namespacing, limits,
    and rules.

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
        """Unstructured key value map that can be used to organize and categorize
    objects. User-provided labels are shared with Google's billing system, so
    they can be used to filter, or break down billing charges by team,
    component, environment, state, etc. For more information, visit
    https://cloud.google.com/resource-manager/docs/creating-managing-labels or
    https://cloud.google.com/run/docs/configuring/labels. Cloud Run API v2
    does not support labels with `run.googleapis.com`, `cloud.googleapis.com`,
    `serving.knative.dev`, or `autoscaling.knative.dev` namespaces, and they
    will be rejected. All system labels in v1 now have a corresponding field
    in v2 Job.

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
    binaryAuthorization = _messages.MessageField('GoogleCloudRunV2BinaryAuthorization', 2)
    client = _messages.StringField(3)
    clientVersion = _messages.StringField(4)
    conditions = _messages.MessageField('GoogleCloudRunV2Condition', 5, repeated=True)
    createTime = _messages.StringField(6)
    creator = _messages.StringField(7)
    deleteTime = _messages.StringField(8)
    etag = _messages.StringField(9)
    executionCount = _messages.IntegerField(10, variant=_messages.Variant.INT32)
    expireTime = _messages.StringField(11)
    generation = _messages.IntegerField(12)
    labels = _messages.MessageField('LabelsValue', 13)
    lastModifier = _messages.StringField(14)
    latestCreatedExecution = _messages.MessageField('GoogleCloudRunV2ExecutionReference', 15)
    launchStage = _messages.EnumField('LaunchStageValueValuesEnum', 16)
    name = _messages.StringField(17)
    observedGeneration = _messages.IntegerField(18)
    reconciling = _messages.BooleanField(19)
    satisfiesPzs = _messages.BooleanField(20)
    template = _messages.MessageField('GoogleCloudRunV2ExecutionTemplate', 21)
    terminalCondition = _messages.MessageField('GoogleCloudRunV2Condition', 22)
    uid = _messages.StringField(23)
    updateTime = _messages.StringField(24)