from __future__ import absolute_import
from apitools.base.py import base_api
from googlecloudsdk.generated_clients.apis.monitoring.v1 import monitoring_v1_messages as messages
def ListMetricsScopesByMonitoredProject(self, request, global_params=None):
    """Returns a list of every Metrics Scope that a specific MonitoredProject has been added to. The metrics scope representing the specified monitored project will always be the first entry in the response.

      Args:
        request: (MonitoringLocationsGlobalMetricsScopesListMetricsScopesByMonitoredProjectRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (ListMetricsScopesByMonitoredProjectResponse) The response message.
      """
    config = self.GetMethodConfig('ListMetricsScopesByMonitoredProject')
    return self._RunMethod(config, request, global_params=global_params)