from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from apitools.base.py import encoding
from googlecloudsdk.api_lib.cloudbuild.v2 import output_util
from googlecloudsdk.core import yaml
from googlecloudsdk.core.resource import custom_printer_base
def InternalPRToTektonPR(self, internal):
    """Convert Tekton yaml file into PipelineRun message."""
    pr = {'metadata': {}, 'spec': {}, 'status': {}}
    if 'name' in internal:
        pr['metadata']['name'] = output_util.ParseName(internal.pop('name'), 'pipelinerun')
    if 'annotations' in internal:
        pr['metadata']['annotations'] = internal.pop('annotations')
    if 'params' in internal:
        pr['spec']['params'] = _TransformParams(internal.pop('params'))
    if 'pipelineSpec' in internal:
        pr['spec']['pipelineSpec'] = _TransformPipelineSpec(internal.pop('pipelineSpec'))
    elif 'pipelineRef' in internal:
        pr['spec']['pipelineRef'] = _TransformPipelineRef(internal.pop('pipelineRef'))
    if 'timeout' in internal:
        pr['spec']['timeout'] = internal.pop('timeout')
    if 'workspaces' in internal:
        pr['spec']['workspaces'] = internal.pop('workspaces')
    if 'conditions' in internal:
        conditions = internal.pop('conditions')
        pr['status']['conditions'] = _TransformConditions(conditions)
    if 'startTime' in internal:
        pr['status']['startTime'] = internal.pop('startTime')
    if 'completionTime' in internal:
        pr['status']['completionTime'] = internal.pop('completionTime')
    if 'resolvedPipelineSpec' in internal:
        rps = internal.pop('resolvedPipelineSpec')
        pr['status']['pipelineSpec'] = _TransformPipelineSpec(rps)
    if 'results' in internal:
        pr['status']['results'] = _TransformPipelineRunResults(internal.pop('results'))
    if 'childReferences' in internal:
        crs = internal.pop('childReferences')
        pr['status']['childReferences'] = crs
    if 'serviceAccount' in internal:
        pr['taskRunTemplate'] = {'serviceAccountName': internal.pop('serviceAccount')}
    return pr