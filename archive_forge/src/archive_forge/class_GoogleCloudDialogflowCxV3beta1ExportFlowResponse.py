from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GoogleCloudDialogflowCxV3beta1ExportFlowResponse(_messages.Message):
    """The response message for Flows.ExportFlow.

  Fields:
    flowContent: Uncompressed raw byte content for flow.
    flowUri: The URI to a file containing the exported flow. This field is
      populated only if `flow_uri` is specified in ExportFlowRequest.
  """
    flowContent = _messages.BytesField(1)
    flowUri = _messages.StringField(2)