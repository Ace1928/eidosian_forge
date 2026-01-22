from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class DataprocProjectsLocationsWorkflowTemplatesInstantiateRequest(_messages.Message):
    """A DataprocProjectsLocationsWorkflowTemplatesInstantiateRequest object.

  Fields:
    instantiateWorkflowTemplateRequest: A InstantiateWorkflowTemplateRequest
      resource to be passed as the request body.
    name: Required. The resource name of the workflow template, as described
      in https://cloud.google.com/apis/design/resource_names. For
      projects.regions.workflowTemplates.instantiate, the resource name of the
      template has the following format:
      projects/{project_id}/regions/{region}/workflowTemplates/{template_id}
      For projects.locations.workflowTemplates.instantiate, the resource name
      of the template has the following format: projects/{project_id}/location
      s/{location}/workflowTemplates/{template_id}
  """
    instantiateWorkflowTemplateRequest = _messages.MessageField('InstantiateWorkflowTemplateRequest', 1)
    name = _messages.StringField(2, required=True)