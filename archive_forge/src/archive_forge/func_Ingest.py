from __future__ import absolute_import
from apitools.base.py import base_api
from googlecloudsdk.generated_clients.apis.healthcare.v1 import healthcare_v1_messages as messages
def Ingest(self, request, global_params=None):
    """Parses and stores an HL7v2 message. This method triggers an asynchronous notification to any Pub/Sub topic configured in Hl7V2Store.Hl7V2NotificationConfig, if the filtering matches the message. If an MLLP adapter is configured to listen to a Pub/Sub topic, the adapter transmits the message when a notification is received. If the method is successful, it generates a response containing an HL7v2 acknowledgment (`ACK`) message. If the method encounters an error, it returns a negative acknowledgment (`NACK`) message. This behavior is suitable for replying to HL7v2 interface systems that expect these acknowledgments.

      Args:
        request: (HealthcareProjectsLocationsDatasetsHl7V2StoresMessagesIngestRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (IngestMessageResponse) The response message.
      """
    config = self.GetMethodConfig('Ingest')
    return self._RunMethod(config, request, global_params=global_params)