from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GoogleCloudRunV2Task(_messages.Message):
    """Task represents a single run of a container to completion.

  Enums:
    ExecutionEnvironmentValueValuesEnum: The execution environment being used
      to host this Task.

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
      https://cloud.google.com/run/docs/configuring/labels

  Fields:
    annotations: Output only. Unstructured key value map that may be set by
      external tools to store and arbitrary metadata. They are not queryable
      and should be preserved when modifying objects.
    completionTime: Output only. Represents time when the Task was completed.
      It is not guaranteed to be set in happens-before order across separate
      operations.
    conditions: Output only. The Condition of this Task, containing its
      readiness status, and detailed error information in case it did not
      reach the desired state.
    containers: Holds the single container that defines the unit of execution
      for this task.
    createTime: Output only. Represents time when the task was created by the
      system. It is not guaranteed to be set in happens-before order across
      separate operations.
    deleteTime: Output only. For a deleted resource, the deletion time. It is
      only populated as a response to a Delete request.
    encryptionKey: Output only. A reference to a customer managed encryption
      key (CMEK) to use to encrypt this container image. For more information,
      go to https://cloud.google.com/run/docs/securing/using-cmek
    etag: Output only. A system-generated fingerprint for this version of the
      resource. May be used to detect modification conflict during updates.
    execution: Output only. The name of the parent Execution.
    executionEnvironment: The execution environment being used to host this
      Task.
    expireTime: Output only. For a deleted resource, the time after which it
      will be permamently deleted. It is only populated as a response to a
      Delete request.
    generation: Output only. A number that monotonically increases every time
      the user modifies the desired state.
    index: Output only. Index of the Task, unique per execution, and beginning
      at 0.
    job: Output only. The name of the parent Job.
    labels: Output only. Unstructured key value map that can be used to
      organize and categorize objects. User-provided labels are shared with
      Google's billing system, so they can be used to filter, or break down
      billing charges by team, component, environment, state, etc. For more
      information, visit https://cloud.google.com/resource-
      manager/docs/creating-managing-labels or
      https://cloud.google.com/run/docs/configuring/labels
    lastAttemptResult: Output only. Result of the last attempt of this Task.
    logUri: Output only. URI where logs for this execution can be found in
      Cloud Console.
    maxRetries: Number of retries allowed per Task, before marking this Task
      failed.
    name: Output only. The unique name of this Task.
    observedGeneration: Output only. The generation of this Task. See comments
      in `Job.reconciling` for additional information on reconciliation
      process in Cloud Run.
    reconciling: Output only. Indicates whether the resource's reconciliation
      is still in progress. See comments in `Job.reconciling` for additional
      information on reconciliation process in Cloud Run.
    retried: Output only. The number of times this Task was retried. Tasks are
      retried when they fail up to the maxRetries limit.
    satisfiesPzs: Output only. Reserved for future use.
    scheduledTime: Output only. Represents time when the task was scheduled to
      run by the system. It is not guaranteed to be set in happens-before
      order across separate operations.
    serviceAccount: Email address of the IAM service account associated with
      the Task of a Job. The service account represents the identity of the
      running task, and determines what permissions the task has. If not
      provided, the task will use the project's default service account.
    startTime: Output only. Represents time when the task started to run. It
      is not guaranteed to be set in happens-before order across separate
      operations.
    timeout: Max allowed time duration the Task may be active before the
      system will actively try to mark it failed and kill associated
      containers. This applies per attempt of a task, meaning each retry can
      run for the full timeout.
    uid: Output only. Server assigned unique identifier for the Task. The
      value is a UUID4 string and guaranteed to remain unchanged until the
      resource is deleted.
    updateTime: Output only. The last-modified time.
    volumes: A list of Volumes to make available to containers.
    vpcAccess: Output only. VPC Access configuration to use for this Task. For
      more information, visit
      https://cloud.google.com/run/docs/configuring/connecting-vpc.
  """

    class ExecutionEnvironmentValueValuesEnum(_messages.Enum):
        """The execution environment being used to host this Task.

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
    labels or https://cloud.google.com/run/docs/configuring/labels

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
    completionTime = _messages.StringField(2)
    conditions = _messages.MessageField('GoogleCloudRunV2Condition', 3, repeated=True)
    containers = _messages.MessageField('GoogleCloudRunV2Container', 4, repeated=True)
    createTime = _messages.StringField(5)
    deleteTime = _messages.StringField(6)
    encryptionKey = _messages.StringField(7)
    etag = _messages.StringField(8)
    execution = _messages.StringField(9)
    executionEnvironment = _messages.EnumField('ExecutionEnvironmentValueValuesEnum', 10)
    expireTime = _messages.StringField(11)
    generation = _messages.IntegerField(12)
    index = _messages.IntegerField(13, variant=_messages.Variant.INT32)
    job = _messages.StringField(14)
    labels = _messages.MessageField('LabelsValue', 15)
    lastAttemptResult = _messages.MessageField('GoogleCloudRunV2TaskAttemptResult', 16)
    logUri = _messages.StringField(17)
    maxRetries = _messages.IntegerField(18, variant=_messages.Variant.INT32)
    name = _messages.StringField(19)
    observedGeneration = _messages.IntegerField(20)
    reconciling = _messages.BooleanField(21)
    retried = _messages.IntegerField(22, variant=_messages.Variant.INT32)
    satisfiesPzs = _messages.BooleanField(23)
    scheduledTime = _messages.StringField(24)
    serviceAccount = _messages.StringField(25)
    startTime = _messages.StringField(26)
    timeout = _messages.StringField(27)
    uid = _messages.StringField(28)
    updateTime = _messages.StringField(29)
    volumes = _messages.MessageField('GoogleCloudRunV2Volume', 30, repeated=True)
    vpcAccess = _messages.MessageField('GoogleCloudRunV2VpcAccess', 31)