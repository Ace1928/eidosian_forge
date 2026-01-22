from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from apitools.base.py import encoding
from googlecloudsdk.api_lib.cloudbuild.v2 import output_util
from googlecloudsdk.core import yaml
from googlecloudsdk.core.resource import custom_printer_base
def _TransformTaskSpec(ts):
    """Convert TaskSpecs into Tekton yaml."""
    task_spec = {}
    if 'params' in ts:
        task_spec['params'] = _TransformParamsSpec(ts.pop('params'))
    if 'steps' in ts:
        task_spec['steps'] = ts.pop('steps')
    if 'stepTemplate' in ts:
        task_spec['stepTemplate'] = ts.pop('stepTemplate')
    if 'results' in ts:
        task_spec['results'] = _TransformTaskResults(ts.pop('results'))
    if 'sidecars' in ts:
        task_spec['sidecars'] = ts.pop('sidecars')
    if 'workspaces' in ts:
        task_spec['workspaces'] = ts.pop('workspaces')
    return task_spec