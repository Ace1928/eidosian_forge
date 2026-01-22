from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GoogleDevtoolsRemotebuildexecutionAdminV1alphaCreateWorkerPoolRequest(_messages.Message):
    """The request used for `CreateWorkerPool`.

  Fields:
    parent: Resource name of the instance in which to create the new worker
      pool. Format: `projects/[PROJECT_ID]/instances/[INSTANCE_ID]`.
    poolId: ID of the created worker pool. A valid pool ID must: be 6-50
      characters long, contain only lowercase letters, digits, hyphens and
      underscores, start with a lowercase letter, and end with a lowercase
      letter or a digit.
    workerPool: Specifies the worker pool to create. The name in the worker
      pool, if specified, is ignored.
  """
    parent = _messages.StringField(1)
    poolId = _messages.StringField(2)
    workerPool = _messages.MessageField('GoogleDevtoolsRemotebuildexecutionAdminV1alphaWorkerPool', 3)