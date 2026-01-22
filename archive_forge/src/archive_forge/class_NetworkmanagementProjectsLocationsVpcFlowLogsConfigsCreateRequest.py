from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class NetworkmanagementProjectsLocationsVpcFlowLogsConfigsCreateRequest(_messages.Message):
    """A NetworkmanagementProjectsLocationsVpcFlowLogsConfigsCreateRequest
  object.

  Fields:
    parent: Required. The parent resource of the VPC Flow Logs configuration
      to create: `projects/{project_id}/locations/global`
    vpcFlowLogsConfig: A VpcFlowLogsConfig resource to be passed as the
      request body.
    vpcFlowLogsConfigId: Required. ID of the VpcFlowLogsConfig.
  """
    parent = _messages.StringField(1, required=True)
    vpcFlowLogsConfig = _messages.MessageField('VpcFlowLogsConfig', 2)
    vpcFlowLogsConfigId = _messages.StringField(3)