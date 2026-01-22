from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class NetworkmanagementProjectsLocationsVpcFlowLogsConfigsGetRequest(_messages.Message):
    """A NetworkmanagementProjectsLocationsVpcFlowLogsConfigsGetRequest object.

  Fields:
    name: Required. `VpcFlowLog` resource name using the form: `projects/{proj
      ect_id}/locations/global/vpcFlowLogsConfigs/{vpc_flow_logs_config}`
  """
    name = _messages.StringField(1, required=True)