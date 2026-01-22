from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class MonitoringLocationsGlobalMetricsScopesProjectsDeleteRequest(_messages.Message):
    """A MonitoringLocationsGlobalMetricsScopesProjectsDeleteRequest object.

  Fields:
    name: Required. The resource name of the MonitoredProject. Example: locati
      ons/global/metricsScopes/{SCOPING_PROJECT_ID_OR_NUMBER}/projects/{MONITO
      RED_PROJECT_ID_OR_NUMBER}Authorization requires the following Google IAM
      (https://cloud.google.com/iam) permissions on both the Metrics Scope and
      on the MonitoredProject: monitoring.metricsScopes.link
  """
    name = _messages.StringField(1, required=True)