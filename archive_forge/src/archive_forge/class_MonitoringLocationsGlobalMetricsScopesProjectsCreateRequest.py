from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class MonitoringLocationsGlobalMetricsScopesProjectsCreateRequest(_messages.Message):
    """A MonitoringLocationsGlobalMetricsScopesProjectsCreateRequest object.

  Fields:
    monitoredProject: A MonitoredProject resource to be passed as the request
      body.
    parent: Required. The resource name of the existing Metrics Scope that
      will monitor this project. Example:
      locations/global/metricsScopes/{SCOPING_PROJECT_ID_OR_NUMBER}
  """
    monitoredProject = _messages.MessageField('MonitoredProject', 1)
    parent = _messages.StringField(2, required=True)