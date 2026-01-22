from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class RunPipelineRequest(_messages.Message):
    """The arguments to the `RunPipeline` method. The requesting user must have
  the `iam.serviceAccounts.actAs` permission for the Cloud Genomics service
  account or the request will fail.

  Messages:
    LabelsValue: User-defined labels to associate with the returned operation.
      These labels are not propagated to any Google Cloud Platform resources
      used by the operation, and can be modified at any time. To associate
      labels with resources created while executing the operation, see the
      appropriate resource message (for example, `VirtualMachine`).

  Fields:
    labels: User-defined labels to associate with the returned operation.
      These labels are not propagated to any Google Cloud Platform resources
      used by the operation, and can be modified at any time. To associate
      labels with resources created while executing the operation, see the
      appropriate resource message (for example, `VirtualMachine`).
    pipeline: Required. The description of the pipeline to run.
    pubSubTopic: The name of an existing Pub/Sub topic. The server will
      publish messages to this topic whenever the status of the operation
      changes. The Genomics Service Agent account must have publisher
      permissions to the specified topic or notifications will not be sent.
  """

    @encoding.MapUnrecognizedFields('additionalProperties')
    class LabelsValue(_messages.Message):
        """User-defined labels to associate with the returned operation. These
    labels are not propagated to any Google Cloud Platform resources used by
    the operation, and can be modified at any time. To associate labels with
    resources created while executing the operation, see the appropriate
    resource message (for example, `VirtualMachine`).

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
    labels = _messages.MessageField('LabelsValue', 1)
    pipeline = _messages.MessageField('Pipeline', 2)
    pubSubTopic = _messages.StringField(3)