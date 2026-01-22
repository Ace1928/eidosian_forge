from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class MonitoringLocationsGlobalMetricsScopesListMetricsScopesByMonitoredProjectRequest(_messages.Message):
    """A MonitoringLocationsGlobalMetricsScopesListMetricsScopesByMonitoredProj
  ectRequest object.

  Fields:
    monitoredResourceContainer: Required. The resource name of the Monitored
      Project being requested. Example:
      projects/{MONITORED_PROJECT_ID_OR_NUMBER}
  """
    monitoredResourceContainer = _messages.StringField(1)