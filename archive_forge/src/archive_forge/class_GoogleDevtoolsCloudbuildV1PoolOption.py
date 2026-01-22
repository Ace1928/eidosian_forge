from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GoogleDevtoolsCloudbuildV1PoolOption(_messages.Message):
    """Details about how a build should be executed on a `WorkerPool`. See
  [running builds in a private
  pool](https://cloud.google.com/build/docs/private-pools/run-builds-in-
  private-pool) for more information.

  Fields:
    name: The `WorkerPool` resource to execute the build on. You must have
      `cloudbuild.workerpools.use` on the project hosting the WorkerPool.
      Format
      projects/{project}/locations/{location}/workerPools/{workerPoolId}
  """
    name = _messages.StringField(1)