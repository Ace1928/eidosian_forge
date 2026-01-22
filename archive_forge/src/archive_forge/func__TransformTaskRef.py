from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from apitools.base.py import encoding
from googlecloudsdk.api_lib.cloudbuild.v2 import output_util
from googlecloudsdk.core import yaml
from googlecloudsdk.core.resource import custom_printer_base
def _TransformTaskRef(tr):
    """Convert TaskRef into Tekton yaml."""
    task_ref = {}
    if 'name' in tr:
        task_ref['name'] = tr.pop('name')
    if 'resolver' in tr:
        task_ref['resolver'] = tr.pop('resolver')
    if 'params' in tr:
        task_ref['params'] = _TransformParams(tr.pop('params'))
    return task_ref