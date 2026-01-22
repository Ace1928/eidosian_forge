from __future__ import absolute_import
from apitools.base.py import base_api
from googlecloudsdk.generated_clients.apis.bigquerydatatransfer.v1 import bigquerydatatransfer_v1_messages as messages
def ScheduleRuns(self, request, global_params=None):
    """Creates transfer runs for a time range [start_time, end_time]. For each date - or whatever granularity the data source supports - in the range, one transfer run is created. Note that runs are created per UTC time in the time range. DEPRECATED: use StartManualTransferRuns instead.

      Args:
        request: (BigquerydatatransferProjectsTransferConfigsScheduleRunsRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (ScheduleTransferRunsResponse) The response message.
      """
    config = self.GetMethodConfig('ScheduleRuns')
    return self._RunMethod(config, request, global_params=global_params)