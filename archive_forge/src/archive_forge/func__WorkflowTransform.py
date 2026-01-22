from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from apitools.base.py import encoding
from googlecloudsdk.api_lib.cloudbuild import cloudbuild_exceptions
from googlecloudsdk.api_lib.cloudbuild.v2 import client_util
from googlecloudsdk.api_lib.cloudbuild.v2 import input_util
from googlecloudsdk.core import yaml
def _WorkflowTransform(workflow):
    """Transform workflow message."""
    _ResourcesTransform(workflow)
    if 'triggers' in workflow:
        workflow['workflowTriggers'] = workflow.pop('triggers')
    for workflow_trigger in workflow.get('workflowTriggers', []):
        input_util.WorkflowTriggerTransform(workflow_trigger, workflow.get('resources', {}))
    for param_spec in workflow.get('params', []):
        input_util.ParamSpecTransform(param_spec)
    pipeline = workflow.pop('pipeline')
    if 'spec' in pipeline:
        workflow['pipelineSpecYaml'] = yaml.dump(pipeline['spec'], round_trip=True)
    elif 'ref' in pipeline:
        input_util.RefTransform(pipeline['ref'])
        workflow['ref'] = pipeline['ref']
    else:
        raise cloudbuild_exceptions.InvalidYamlError('PipelineSpec or PipelineRef is required.')
    for workspace_binding in workflow.get('workspaces', []):
        _WorkspaceBindingTransform(workspace_binding)
    if 'options' in workflow and 'status' in workflow['options']:
        popped_status = workflow['options'].pop('status')
        workflow['options']['statusUpdateOptions'] = popped_status
    for option in _WORKFLOW_OPTIONS_ENUMS:
        input_util.SetDictDottedKeyUpperCase(workflow, option)