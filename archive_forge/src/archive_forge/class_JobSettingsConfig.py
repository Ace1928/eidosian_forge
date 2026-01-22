from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class JobSettingsConfig(_messages.Message):
    """Message for Cloud Run Job settings config. Next tag: 8

  Messages:
    EnvVarsValue: Key-value pairs to set as environment variables. Note that
      integration bindings will add/update the list of final env vars that are
      deployed to a job.

  Fields:
    args: Comma-separated arguments passed to the command run by the container
      image.
    cmd: Entrypoint for the container image.
    envVars: Key-value pairs to set as environment variables. Note that
      integration bindings will add/update the list of final env vars that are
      deployed to a job.
    image: The container image to deploy the job with.
    maxRetries: Number of times a task is allowed to restart in case of
      failure before being failed permanently. This applies per-task, not per-
      job. If set to 0, tasks will only run once and never be retried on
      failure. Default value is 3.
    parallelism: Number of tasks that may run concurrently. Must be less than
      or equal to the number of tasks. When the job is run, if this field is 0
      or unset, the maximum possible value will be used for that execution.
      Default is unset.
    taskCount: Specifies the desired number of tasks the execution should run.
      Setting to 1 means that parallelism is limited to 1 and the success of
      that task signals the success of the execution. Default value is 1.
  """

    @encoding.MapUnrecognizedFields('additionalProperties')
    class EnvVarsValue(_messages.Message):
        """Key-value pairs to set as environment variables. Note that integration
    bindings will add/update the list of final env vars that are deployed to a
    job.

    Messages:
      AdditionalProperty: An additional property for a EnvVarsValue object.

    Fields:
      additionalProperties: Additional properties of type EnvVarsValue
    """

        class AdditionalProperty(_messages.Message):
            """An additional property for a EnvVarsValue object.

      Fields:
        key: Name of the additional property.
        value: A string attribute.
      """
            key = _messages.StringField(1)
            value = _messages.StringField(2)
        additionalProperties = _messages.MessageField('AdditionalProperty', 1, repeated=True)
    args = _messages.StringField(1, repeated=True)
    cmd = _messages.StringField(2, repeated=True)
    envVars = _messages.MessageField('EnvVarsValue', 3)
    image = _messages.StringField(4)
    maxRetries = _messages.IntegerField(5, variant=_messages.Variant.INT32)
    parallelism = _messages.IntegerField(6, variant=_messages.Variant.INT32)
    taskCount = _messages.IntegerField(7, variant=_messages.Variant.INT32)