from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class CloudbuildProjectsLocationsTaskRunsPatchRequest(_messages.Message):
    """A CloudbuildProjectsLocationsTaskRunsPatchRequest object.

  Fields:
    allowMissing: Optional. If set to true, and the TaskRun is not found, a
      new TaskRun will be created. In this situation, `update_mask` is
      ignored.
    name: Output only. The 'TaskRun' name with format:
      `projects/{project}/locations/{location}/taskRuns/{task_run}`
    taskRun: A TaskRun resource to be passed as the request body.
    updateMask: Required. The list of fields to be updated.
    validateOnly: Optional. When true, the query is validated only, but not
      executed.
  """
    allowMissing = _messages.BooleanField(1)
    name = _messages.StringField(2, required=True)
    taskRun = _messages.MessageField('TaskRun', 3)
    updateMask = _messages.StringField(4)
    validateOnly = _messages.BooleanField(5)