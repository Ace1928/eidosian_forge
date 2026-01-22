from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class MonitoringProjectsDashboardsPatchRequest(_messages.Message):
    """A MonitoringProjectsDashboardsPatchRequest object.

  Fields:
    dashboard: A Dashboard resource to be passed as the request body.
    name: Identifier. The resource name of the dashboard.
    validateOnly: If set, validate the request and preview the review, but do
      not actually save it.
  """
    dashboard = _messages.MessageField('Dashboard', 1)
    name = _messages.StringField(2, required=True)
    validateOnly = _messages.BooleanField(3)