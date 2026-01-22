from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals
def GetSuccessMessageForWorkerDeploy(worker):
    """Returns a user message for a successful synchronous deploy.

  TODO(b/322180968): Once Worker API is ready, replace Service related
  references.
  Args:
    worker: googlecloudsdk.api_lib.run.service.Service, Deployed service for
      which to build a success message.
  """
    latest_ready = worker.status.latestReadyRevisionName
    msg = 'Worker [{{bold}}{worker}{{reset}}] revision [{{bold}}{rev}{{reset}}] has been deployed.'
    return msg.format(worker=worker.name, rev=latest_ready)