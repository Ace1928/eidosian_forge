from libcloud.common.google import GoogleResponse, GoogleBaseConnection, GoogleOAuth2Credential
from libcloud.container.providers import Provider
from libcloud.container.drivers.kubernetes import KubernetesContainerDriver
class GKEConnection(GoogleBaseConnection):
    """
    Connection class for the GKE driver.

    GKEConnection extends :class:`google.GoogleBaseConnection` for 3 reasons:
      1. modify request_path for GKE URI.
      2. Implement gce_params functionality described below.
      3. Add request_aggregated_items method for making aggregated API calls.

    """
    host = 'container.googleapis.com'
    responseCls = GKEResponse

    def __init__(self, user_id, key, secure, auth_type=None, credential_file=None, project=None, **kwargs):
        super().__init__(user_id, key, secure=secure, auth_type=auth_type, credential_file=credential_file, **kwargs)
        self.request_path = '/{}/projects/{}'.format(API_VERSION, project)
        self.gke_params = {}

    def pre_connect_hook(self, params, headers):
        """
        Update URL parameters with values from self.gke_params.

        @inherits: :class:`GoogleBaseConnection.pre_connect_hook`
        """
        params, headers = super().pre_connect_hook(params, headers)
        if self.gke_params:
            params.update(self.gke_params)
        return (params, headers)

    def request(self, *args, **kwargs):
        """
        Perform request then do GKE-specific processing of URL params.

        @inherits: :class:`GoogleBaseConnection.request`
        """
        response = super().request(*args, **kwargs)
        if self.gke_params:
            if 'nextPageToken' in response.object:
                self.gke_params['pageToken'] = response.object['nextPageToken']
            elif 'pageToken' in self.gke_params:
                del self.gke_params['pageToken']
            self.gke_params = None
        return response