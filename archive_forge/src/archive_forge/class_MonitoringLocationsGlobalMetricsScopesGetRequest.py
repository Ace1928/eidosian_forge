from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class MonitoringLocationsGlobalMetricsScopesGetRequest(_messages.Message):
    """A MonitoringLocationsGlobalMetricsScopesGetRequest object.

  Fields:
    name: Required. The resource name of the Metrics Scope. Example:
      locations/global/metricsScopes/{SCOPING_PROJECT_ID_OR_NUMBER}
  """
    name = _messages.StringField(1, required=True)