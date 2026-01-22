from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GoogleCloudAiplatformV1beta1IndexPrivateEndpoints(_messages.Message):
    """IndexPrivateEndpoints proto is used to provide paths for users to send
  requests via private endpoints (e.g. private service access, private service
  connect). To send request via private service access, use
  match_grpc_address. To send request via private service connect, use
  service_attachment.

  Fields:
    matchGrpcAddress: Output only. The ip address used to send match gRPC
      requests.
    pscAutomatedEndpoints: Output only. PscAutomatedEndpoints is populated if
      private service connect is enabled if PscAutomatedConfig is set.
    serviceAttachment: Output only. The name of the service attachment
      resource. Populated if private service connect is enabled.
  """
    matchGrpcAddress = _messages.StringField(1)
    pscAutomatedEndpoints = _messages.MessageField('GoogleCloudAiplatformV1beta1PscAutomatedEndpoints', 2, repeated=True)
    serviceAttachment = _messages.StringField(3)