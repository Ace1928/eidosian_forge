from __future__ import absolute_import
from apitools.base.py import base_api
from samples.bigquery_sample.bigquery_v2 import bigquery_v2_messages as messages
def Cancel(self, request, global_params=None):
    """Requests that a job be cancelled. This call will return immediately, and the client will need to poll for the job status to see if the cancel completed successfully. Cancelled jobs may still incur costs.

      Args:
        request: (BigqueryJobsCancelRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (JobCancelResponse) The response message.
      """
    config = self.GetMethodConfig('Cancel')
    return self._RunMethod(config, request, global_params=global_params)