from __future__ import absolute_import
from apitools.base.py import base_api
from googlecloudsdk.generated_clients.apis.healthcare.v1beta1 import healthcare_v1beta1_messages as messages
def Resource_incoming_references(self, request, global_params=None):
    """Lists all the resources that directly refer to the given target FHIR resource. Can also support the case when the target resource doesn't exist, for example, if the target has been deleted. On success, the response body contains a Bundle with type `searchset`, where each entry in the Bundle contains the full content of the resource. If the operation fails, an `OperationOutcome` is returned describing the failure. If the request cannot be mapped to a valid API method on a FHIR store, a generic Google Cloud error might be returned instead.

      Args:
        request: (HealthcareProjectsLocationsDatasetsFhirStoresFhirResourceIncomingReferencesRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (HttpBody) The response message.
      """
    config = self.GetMethodConfig('Resource_incoming_references')
    return self._RunMethod(config, request, global_params=global_params)