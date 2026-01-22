from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GoogleCloudRunV2RevisionTemplate(_messages.Message):
    """RevisionTemplate describes the data a revision should have when created
  from a template.

  Enums:
    ExecutionEnvironmentValueValuesEnum: Optional. The sandbox environment to
      host this Revision.

  Messages:
    AnnotationsValue: Optional. Unstructured key value map that may be set by
      external tools to store and arbitrary metadata. They are not queryable
      and should be preserved when modifying objects. Cloud Run API v2 does
      not support annotations with `run.googleapis.com`,
      `cloud.googleapis.com`, `serving.knative.dev`, or
      `autoscaling.knative.dev` namespaces, and they will be rejected. All
      system annotations in v1 now have a corresponding field in v2
      RevisionTemplate. This field follows Kubernetes annotations'
      namespacing, limits, and rules.
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
      system labels in v1 now have a corresponding field in v2
      RevisionTemplate.

  Fields:
    annotations: Optional. Unstructured key value map that may be set by
      external tools to store and arbitrary metadata. They are not queryable
      and should be preserved when modifying objects. Cloud Run API v2 does
      not support annotations with `run.googleapis.com`,
      `cloud.googleapis.com`, `serving.knative.dev`, or
      `autoscaling.knative.dev` namespaces, and they will be rejected. All
      system annotations in v1 now have a corresponding field in v2
      RevisionTemplate. This field follows Kubernetes annotations'
      namespacing, limits, and rules.
    containers: Holds the single container that defines the unit of execution
      for this Revision.
    encryptionKey: A reference to a customer managed encryption key (CMEK) to
      use to encrypt this container image. For more information, go to
      https://cloud.google.com/run/docs/securing/using-cmek
    executionEnvironment: Optional. The sandbox environment to host this
      Revision.
    healthCheckDisabled: Optional. Disables health checking containers during
      deployment.
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
      system labels in v1 now have a corresponding field in v2
      RevisionTemplate.
    maxInstanceRequestConcurrency: Optional. Sets the maximum number of
      requests that each serving instance can receive.
    revision: Optional. The unique name for the revision. If this field is
      omitted, it will be automatically generated based on the Service name.
    scaling: Optional. Scaling settings for this Revision.
    serviceAccount: Optional. Email address of the IAM service account
      associated with the revision of the service. The service account
      represents the identity of the running revision, and determines what
      permissions the revision has. If not provided, the revision will use the
      project's default service account.
    sessionAffinity: Optional. Enable session affinity.
    timeout: Optional. Max allowed time for an instance to respond to a
      request.
    volumes: Optional. A list of Volumes to make available to containers.
    vpcAccess: Optional. VPC Access configuration to use for this Revision.
      For more information, visit
      https://cloud.google.com/run/docs/configuring/connecting-vpc.
  """

    class ExecutionEnvironmentValueValuesEnum(_messages.Enum):
        """Optional. The sandbox environment to host this Revision.

    Values:
      EXECUTION_ENVIRONMENT_UNSPECIFIED: Unspecified
      EXECUTION_ENVIRONMENT_GEN1: Uses the First Generation environment.
      EXECUTION_ENVIRONMENT_GEN2: Uses Second Generation environment.
    """
        EXECUTION_ENVIRONMENT_UNSPECIFIED = 0
        EXECUTION_ENVIRONMENT_GEN1 = 1
        EXECUTION_ENVIRONMENT_GEN2 = 2

    @encoding.MapUnrecognizedFields('additionalProperties')
    class AnnotationsValue(_messages.Message):
        """Optional. Unstructured key value map that may be set by external tools
    to store and arbitrary metadata. They are not queryable and should be
    preserved when modifying objects. Cloud Run API v2 does not support
    annotations with `run.googleapis.com`, `cloud.googleapis.com`,
    `serving.knative.dev`, or `autoscaling.knative.dev` namespaces, and they
    will be rejected. All system annotations in v1 now have a corresponding
    field in v2 RevisionTemplate. This field follows Kubernetes annotations'
    namespacing, limits, and rules.

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
    in v2 RevisionTemplate.

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
    containers = _messages.MessageField('GoogleCloudRunV2Container', 2, repeated=True)
    encryptionKey = _messages.StringField(3)
    executionEnvironment = _messages.EnumField('ExecutionEnvironmentValueValuesEnum', 4)
    healthCheckDisabled = _messages.BooleanField(5)
    labels = _messages.MessageField('LabelsValue', 6)
    maxInstanceRequestConcurrency = _messages.IntegerField(7, variant=_messages.Variant.INT32)
    revision = _messages.StringField(8)
    scaling = _messages.MessageField('GoogleCloudRunV2RevisionScaling', 9)
    serviceAccount = _messages.StringField(10)
    sessionAffinity = _messages.BooleanField(11)
    timeout = _messages.StringField(12)
    volumes = _messages.MessageField('GoogleCloudRunV2Volume', 13, repeated=True)
    vpcAccess = _messages.MessageField('GoogleCloudRunV2VpcAccess', 14)