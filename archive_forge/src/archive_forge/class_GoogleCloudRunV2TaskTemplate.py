from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GoogleCloudRunV2TaskTemplate(_messages.Message):
    """TaskTemplate describes the data a task should have when created from a
  template.

  Enums:
    ExecutionEnvironmentValueValuesEnum: Optional. The execution environment
      being used to host this Task.

  Fields:
    containers: Holds the single container that defines the unit of execution
      for this task.
    encryptionKey: A reference to a customer managed encryption key (CMEK) to
      use to encrypt this container image. For more information, go to
      https://cloud.google.com/run/docs/securing/using-cmek
    executionEnvironment: Optional. The execution environment being used to
      host this Task.
    maxRetries: Number of retries allowed per Task, before marking this Task
      failed. Defaults to 3.
    serviceAccount: Optional. Email address of the IAM service account
      associated with the Task of a Job. The service account represents the
      identity of the running task, and determines what permissions the task
      has. If not provided, the task will use the project's default service
      account.
    timeout: Optional. Max allowed time duration the Task may be active before
      the system will actively try to mark it failed and kill associated
      containers. This applies per attempt of a task, meaning each retry can
      run for the full timeout. Defaults to 600 seconds.
    volumes: Optional. A list of Volumes to make available to containers.
    vpcAccess: Optional. VPC Access configuration to use for this Task. For
      more information, visit
      https://cloud.google.com/run/docs/configuring/connecting-vpc.
  """

    class ExecutionEnvironmentValueValuesEnum(_messages.Enum):
        """Optional. The execution environment being used to host this Task.

    Values:
      EXECUTION_ENVIRONMENT_UNSPECIFIED: Unspecified
      EXECUTION_ENVIRONMENT_GEN1: Uses the First Generation environment.
      EXECUTION_ENVIRONMENT_GEN2: Uses Second Generation environment.
    """
        EXECUTION_ENVIRONMENT_UNSPECIFIED = 0
        EXECUTION_ENVIRONMENT_GEN1 = 1
        EXECUTION_ENVIRONMENT_GEN2 = 2
    containers = _messages.MessageField('GoogleCloudRunV2Container', 1, repeated=True)
    encryptionKey = _messages.StringField(2)
    executionEnvironment = _messages.EnumField('ExecutionEnvironmentValueValuesEnum', 3)
    maxRetries = _messages.IntegerField(4, variant=_messages.Variant.INT32)
    serviceAccount = _messages.StringField(5)
    timeout = _messages.StringField(6)
    volumes = _messages.MessageField('GoogleCloudRunV2Volume', 7, repeated=True)
    vpcAccess = _messages.MessageField('GoogleCloudRunV2VpcAccess', 8)