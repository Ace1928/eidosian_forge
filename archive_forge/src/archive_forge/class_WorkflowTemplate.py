from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class WorkflowTemplate(_messages.Message):
    """A Dataproc workflow template resource.

  Messages:
    LabelsValue: Optional. The labels to associate with this template. These
      labels will be propagated to all jobs and clusters created by the
      workflow instance.Label keys must contain 1 to 63 characters, and must
      conform to RFC 1035 (https://www.ietf.org/rfc/rfc1035.txt).Label values
      may be empty, but, if present, must contain 1 to 63 characters, and must
      conform to RFC 1035 (https://www.ietf.org/rfc/rfc1035.txt).No more than
      32 labels can be associated with a template.

  Fields:
    createTime: Output only. The time template was created.
    dagTimeout: Optional. Timeout duration for the DAG of jobs, expressed in
      seconds (see JSON representation of duration
      (https://developers.google.com/protocol-buffers/docs/proto3#json)). The
      timeout duration must be from 10 minutes ("600s") to 24 hours
      ("86400s"). The timer begins when the first job is submitted. If the
      workflow is running at the end of the timeout period, any remaining jobs
      are cancelled, the workflow is ended, and if the workflow was running on
      a managed cluster, the cluster is deleted.
    encryptionConfig: Optional. Encryption settings for encrypting workflow
      template job arguments.
    id: A string attribute.
    jobs: Required. The Directed Acyclic Graph of Jobs to submit.
    labels: Optional. The labels to associate with this template. These labels
      will be propagated to all jobs and clusters created by the workflow
      instance.Label keys must contain 1 to 63 characters, and must conform to
      RFC 1035 (https://www.ietf.org/rfc/rfc1035.txt).Label values may be
      empty, but, if present, must contain 1 to 63 characters, and must
      conform to RFC 1035 (https://www.ietf.org/rfc/rfc1035.txt).No more than
      32 labels can be associated with a template.
    name: Output only. The resource name of the workflow template, as
      described in https://cloud.google.com/apis/design/resource_names. For
      projects.regions.workflowTemplates, the resource name of the template
      has the following format:
      projects/{project_id}/regions/{region}/workflowTemplates/{template_id}
      For projects.locations.workflowTemplates, the resource name of the
      template has the following format: projects/{project_id}/locations/{loca
      tion}/workflowTemplates/{template_id}
    parameters: Optional. Template parameters whose values are substituted
      into the template. Values for parameters must be provided when the
      template is instantiated.
    placement: Required. WorkflowTemplate scheduling information.
    updateTime: Output only. The time template was last updated.
    version: Optional. Used to perform a consistent read-modify-write.This
      field should be left blank for a CreateWorkflowTemplate request. It is
      required for an UpdateWorkflowTemplate request, and must match the
      current server version. A typical update template flow would fetch the
      current template with a GetWorkflowTemplate request, which will return
      the current template with the version field filled in with the current
      server version. The user updates other fields in the template, then
      returns it as part of the UpdateWorkflowTemplate request.
  """

    @encoding.MapUnrecognizedFields('additionalProperties')
    class LabelsValue(_messages.Message):
        """Optional. The labels to associate with this template. These labels
    will be propagated to all jobs and clusters created by the workflow
    instance.Label keys must contain 1 to 63 characters, and must conform to
    RFC 1035 (https://www.ietf.org/rfc/rfc1035.txt).Label values may be empty,
    but, if present, must contain 1 to 63 characters, and must conform to RFC
    1035 (https://www.ietf.org/rfc/rfc1035.txt).No more than 32 labels can be
    associated with a template.

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
    createTime = _messages.StringField(1)
    dagTimeout = _messages.StringField(2)
    encryptionConfig = _messages.MessageField('GoogleCloudDataprocV1WorkflowTemplateEncryptionConfig', 3)
    id = _messages.StringField(4)
    jobs = _messages.MessageField('OrderedJob', 5, repeated=True)
    labels = _messages.MessageField('LabelsValue', 6)
    name = _messages.StringField(7)
    parameters = _messages.MessageField('TemplateParameter', 8, repeated=True)
    placement = _messages.MessageField('WorkflowTemplatePlacement', 9)
    updateTime = _messages.StringField(10)
    version = _messages.IntegerField(11, variant=_messages.Variant.INT32)