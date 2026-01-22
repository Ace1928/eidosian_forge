from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class MonitoringProjectsMonitoredResourceDescriptorsGetRequest(_messages.Message):
    """A MonitoringProjectsMonitoredResourceDescriptorsGetRequest object.

  Fields:
    name: Required. The monitored resource descriptor to get. The format is: p
      rojects/[PROJECT_ID_OR_NUMBER]/monitoredResourceDescriptors/[RESOURCE_TY
      PE] The [RESOURCE_TYPE] is a predefined type, such as cloudsql_database.
  """
    name = _messages.StringField(1, required=True)