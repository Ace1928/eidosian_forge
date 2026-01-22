from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from apitools.base.py import encoding
from googlecloudsdk.api_lib.cloudbuild import cloudbuild_exceptions
from googlecloudsdk.api_lib.cloudbuild.v2 import client_util
from googlecloudsdk.api_lib.cloudbuild.v2 import input_util
from googlecloudsdk.core import yaml
def _PipelineSpecTransform(pipeline_spec):
    """Transform pipeline spec message."""
    for pipeline_task in pipeline_spec.get('tasks', []):
        _PipelineTaskTransform(pipeline_task)
    for param_spec in pipeline_spec.get('params', []):
        input_util.ParamSpecTransform(param_spec)
    if 'finally' in pipeline_spec:
        finally_tasks = pipeline_spec.pop('finally')
        for task in finally_tasks:
            _PipelineTaskTransform(task)
        pipeline_spec['finallyTasks'] = finally_tasks