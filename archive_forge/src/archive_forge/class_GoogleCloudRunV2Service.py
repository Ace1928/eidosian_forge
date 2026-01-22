from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GoogleCloudRunV2Service(_messages.Message):
    """Service acts as a top-level container that manages a set of
  configurations and revision templates which implement a network service.
  Service exists to provide a singular abstraction which can be access
  controlled, reasoned about, and which encapsulates software lifecycle
  decisions such as rollout policy and team resource ownership.

  Enums:
    IngressValueValuesEnum: Optional. Provides the ingress settings for this
      Service. On output, returns the currently observed ingress settings, or
      INGRESS_TRAFFIC_UNSPECIFIED if no revision is active.
    LaunchStageValueValuesEnum: Optional. The launch stage as defined by
      [Google Cloud Platform Launch
      Stages](https://cloud.google.com/terms/launch-stages). Cloud Run
      supports `ALPHA`, `BETA`, and `GA`. If no value is specified, GA is
      assumed. Set the launch stage to a preview stage on input to allow use
      of preview features in that stage. On read (or output), describes
      whether the resource uses preview features. For example, if ALPHA is
      provided as input, but only BETA and GA-level features are used, this
      field will be BETA on output.

  Messages:
    AnnotationsValue: Optional. Unstructured key value map that may be set by
      external tools to store and arbitrary metadata. They are not queryable
      and should be preserved when modifying objects. Cloud Run API v2 does
      not support annotations with `run.googleapis.com`,
      `cloud.googleapis.com`, `serving.knative.dev`, or
      `autoscaling.knative.dev` namespaces, and they will be rejected in new
      resources. All system annotations in v1 now have a corresponding field
      in v2 Service. This field follows Kubernetes annotations' namespacing,
      limits, and rules.
    LabelsValue: Optional. Unstructured key value map that can be used to
      organize and categorize objects. User-provided labels are shared with
      Google's billing system, so they can be used to filter, or break down
      billing charges by team, component, environment, state, etc. For more
      information, visit https://cloud.google.com/resource-
      manager/docs/creating-managing-labels or
      https://cloud.google.com/run/docs/configuring/labels. Cloud Run API v2
      does not support labels with `run.googleapis.com`,
      `cloud.googleapis.com`, `serving.knative.dev`, or
      `autoscaling.knative.dev` namespaces, and they will be rejected. All
      system labels in v1 now have a corresponding field in v2 Service.

  Fields:
    annotations: Optional. Unstructured key value map that may be set by
      external tools to store and arbitrary metadata. They are not queryable
      and should be preserved when modifying objects. Cloud Run API v2 does
      not support annotations with `run.googleapis.com`,
      `cloud.googleapis.com`, `serving.knative.dev`, or
      `autoscaling.knative.dev` namespaces, and they will be rejected in new
      resources. All system annotations in v1 now have a corresponding field
      in v2 Service. This field follows Kubernetes annotations' namespacing,
      limits, and rules.
    binaryAuthorization: Optional. Settings for the Binary Authorization
      feature.
    client: Arbitrary identifier for the API client.
    clientVersion: Arbitrary version identifier for the API client.
    conditions: Output only. The Conditions of all other associated sub-
      resources. They contain additional diagnostics information in case the
      Service does not reach its Serving state. See comments in `reconciling`
      for additional information on reconciliation process in Cloud Run.
    createTime: Output only. The creation time.
    creator: Output only. Email address of the authenticated creator.
    customAudiences: One or more custom audiences that you want this service
      to support. Specify each custom audience as the full URL in a string.
      The custom audiences are encoded in the token and used to authenticate
      requests. For more information, see
      https://cloud.google.com/run/docs/configuring/custom-audiences.
    defaultUriDisabled: Optional. Disables public resolution of the default
      URI of this service.
    deleteTime: Output only. The deletion time.
    description: User-provided description of the Service. This field
      currently has a 512-character limit.
    etag: Output only. A system-generated fingerprint for this version of the
      resource. May be used to detect modification conflict during updates.
    expireTime: Output only. For a deleted resource, the time after which it
      will be permamently deleted.
    generation: Output only. A number that monotonically increases every time
      the user modifies the desired state. Please note that unlike v1, this is
      an int64 value. As with most Google APIs, its JSON representation will
      be a `string` instead of an `integer`.
    ingress: Optional. Provides the ingress settings for this Service. On
      output, returns the currently observed ingress settings, or
      INGRESS_TRAFFIC_UNSPECIFIED if no revision is active.
    labels: Optional. Unstructured key value map that can be used to organize
      and categorize objects. User-provided labels are shared with Google's
      billing system, so they can be used to filter, or break down billing
      charges by team, component, environment, state, etc. For more
      information, visit https://cloud.google.com/resource-
      manager/docs/creating-managing-labels or
      https://cloud.google.com/run/docs/configuring/labels. Cloud Run API v2
      does not support labels with `run.googleapis.com`,
      `cloud.googleapis.com`, `serving.knative.dev`, or
      `autoscaling.knative.dev` namespaces, and they will be rejected. All
      system labels in v1 now have a corresponding field in v2 Service.
    lastModifier: Output only. Email address of the last authenticated
      modifier.
    latestCreatedRevision: Output only. Name of the last created revision. See
      comments in `reconciling` for additional information on reconciliation
      process in Cloud Run.
    latestReadyRevision: Output only. Name of the latest revision that is
      serving traffic. See comments in `reconciling` for additional
      information on reconciliation process in Cloud Run.
    launchStage: Optional. The launch stage as defined by [Google Cloud
      Platform Launch Stages](https://cloud.google.com/terms/launch-stages).
      Cloud Run supports `ALPHA`, `BETA`, and `GA`. If no value is specified,
      GA is assumed. Set the launch stage to a preview stage on input to allow
      use of preview features in that stage. On read (or output), describes
      whether the resource uses preview features. For example, if ALPHA is
      provided as input, but only BETA and GA-level features are used, this
      field will be BETA on output.
    name: The fully qualified name of this Service. In CreateServiceRequest,
      this field is ignored, and instead composed from
      CreateServiceRequest.parent and CreateServiceRequest.service_id. Format:
      projects/{project}/locations/{location}/services/{service_id}
    observedGeneration: Output only. The generation of this Service currently
      serving traffic. See comments in `reconciling` for additional
      information on reconciliation process in Cloud Run. Please note that
      unlike v1, this is an int64 value. As with most Google APIs, its JSON
      representation will be a `string` instead of an `integer`.
    reconciling: Output only. Returns true if the Service is currently being
      acted upon by the system to bring it into the desired state. When a new
      Service is created, or an existing one is updated, Cloud Run will
      asynchronously perform all necessary steps to bring the Service to the
      desired serving state. This process is called reconciliation. While
      reconciliation is in process, `observed_generation`,
      `latest_ready_revison`, `traffic_statuses`, and `uri` will have
      transient values that might mismatch the intended state: Once
      reconciliation is over (and this field is false), there are two possible
      outcomes: reconciliation succeeded and the serving state matches the
      Service, or there was an error, and reconciliation failed. This state
      can be found in `terminal_condition.state`. If reconciliation succeeded,
      the following fields will match: `traffic` and `traffic_statuses`,
      `observed_generation` and `generation`, `latest_ready_revision` and
      `latest_created_revision`. If reconciliation failed, `traffic_statuses`,
      `observed_generation`, and `latest_ready_revision` will have the state
      of the last serving revision, or empty for newly created Services.
      Additional information on the failure can be found in
      `terminal_condition` and `conditions`.
    satisfiesPzs: Output only. Reserved for future use.
    scaling: Optional. Specifies service-level scaling settings
    template: Required. The template used to create revisions for this
      Service.
    terminalCondition: Output only. The Condition of this Service, containing
      its readiness status, and detailed error information in case it did not
      reach a serving state. See comments in `reconciling` for additional
      information on reconciliation process in Cloud Run.
    traffic: Optional. Specifies how to distribute traffic over a collection
      of Revisions belonging to the Service. If traffic is empty or not
      provided, defaults to 100% traffic to the latest `Ready` Revision.
    trafficStatuses: Output only. Detailed status information for
      corresponding traffic targets. See comments in `reconciling` for
      additional information on reconciliation process in Cloud Run.
    uid: Output only. Server assigned unique identifier for the trigger. The
      value is a UUID4 string and guaranteed to remain unchanged until the
      resource is deleted.
    updateTime: Output only. The last-modified time.
    uri: Output only. The main URI in which this Service is serving traffic.
  """

    class IngressValueValuesEnum(_messages.Enum):
        """Optional. Provides the ingress settings for this Service. On output,
    returns the currently observed ingress settings, or
    INGRESS_TRAFFIC_UNSPECIFIED if no revision is active.

    Values:
      INGRESS_TRAFFIC_UNSPECIFIED: Unspecified
      INGRESS_TRAFFIC_ALL: All inbound traffic is allowed.
      INGRESS_TRAFFIC_INTERNAL_ONLY: Only internal traffic is allowed.
      INGRESS_TRAFFIC_INTERNAL_LOAD_BALANCER: Both internal and Google Cloud
        Load Balancer traffic is allowed.
      INGRESS_TRAFFIC_NONE: No ingress traffic is allowed.
    """
        INGRESS_TRAFFIC_UNSPECIFIED = 0
        INGRESS_TRAFFIC_ALL = 1
        INGRESS_TRAFFIC_INTERNAL_ONLY = 2
        INGRESS_TRAFFIC_INTERNAL_LOAD_BALANCER = 3
        INGRESS_TRAFFIC_NONE = 4

    class LaunchStageValueValuesEnum(_messages.Enum):
        """Optional. The launch stage as defined by [Google Cloud Platform Launch
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
        """Optional. Unstructured key value map that may be set by external tools
    to store and arbitrary metadata. They are not queryable and should be
    preserved when modifying objects. Cloud Run API v2 does not support
    annotations with `run.googleapis.com`, `cloud.googleapis.com`,
    `serving.knative.dev`, or `autoscaling.knative.dev` namespaces, and they
    will be rejected in new resources. All system annotations in v1 now have a
    corresponding field in v2 Service. This field follows Kubernetes
    annotations' namespacing, limits, and rules.

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
        """Optional. Unstructured key value map that can be used to organize and
    categorize objects. User-provided labels are shared with Google's billing
    system, so they can be used to filter, or break down billing charges by
    team, component, environment, state, etc. For more information, visit
    https://cloud.google.com/resource-manager/docs/creating-managing-labels or
    https://cloud.google.com/run/docs/configuring/labels. Cloud Run API v2
    does not support labels with `run.googleapis.com`, `cloud.googleapis.com`,
    `serving.knative.dev`, or `autoscaling.knative.dev` namespaces, and they
    will be rejected. All system labels in v1 now have a corresponding field
    in v2 Service.

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
    customAudiences = _messages.StringField(8, repeated=True)
    defaultUriDisabled = _messages.BooleanField(9)
    deleteTime = _messages.StringField(10)
    description = _messages.StringField(11)
    etag = _messages.StringField(12)
    expireTime = _messages.StringField(13)
    generation = _messages.IntegerField(14)
    ingress = _messages.EnumField('IngressValueValuesEnum', 15)
    labels = _messages.MessageField('LabelsValue', 16)
    lastModifier = _messages.StringField(17)
    latestCreatedRevision = _messages.StringField(18)
    latestReadyRevision = _messages.StringField(19)
    launchStage = _messages.EnumField('LaunchStageValueValuesEnum', 20)
    name = _messages.StringField(21)
    observedGeneration = _messages.IntegerField(22)
    reconciling = _messages.BooleanField(23)
    satisfiesPzs = _messages.BooleanField(24)
    scaling = _messages.MessageField('GoogleCloudRunV2ServiceScaling', 25)
    template = _messages.MessageField('GoogleCloudRunV2RevisionTemplate', 26)
    terminalCondition = _messages.MessageField('GoogleCloudRunV2Condition', 27)
    traffic = _messages.MessageField('GoogleCloudRunV2TrafficTarget', 28, repeated=True)
    trafficStatuses = _messages.MessageField('GoogleCloudRunV2TrafficTargetStatus', 29, repeated=True)
    uid = _messages.StringField(30)
    updateTime = _messages.StringField(31)
    uri = _messages.StringField(32)