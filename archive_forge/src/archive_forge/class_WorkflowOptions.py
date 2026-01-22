from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class WorkflowOptions(_messages.Message):
    """Workflow runs can be modified through several Workflow options.

  Fields:
    executionEnvironment: Contains the workerpool.
    provenance: Optional. Provenance configuration.
    security: Optional. Security configuration.
    statusUpdateOptions: How/where status on the workflow is posted.
    timeouts: Time after which the Pipeline times out. Currently three keys
      are accepted in the map pipeline, tasks and finally with
      Timeouts.pipeline >= Timeouts.tasks + Timeouts.finally
    worker: Optional. Worker config.
  """
    executionEnvironment = _messages.MessageField('ExecutionEnvironment', 1)
    provenance = _messages.MessageField('Provenance', 2)
    security = _messages.MessageField('Security', 3)
    statusUpdateOptions = _messages.MessageField('WorkflowStatusUpdateOptions', 4)
    timeouts = _messages.MessageField('TimeoutFields', 5)
    worker = _messages.MessageField('Worker', 6)