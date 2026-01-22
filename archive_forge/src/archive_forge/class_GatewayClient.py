from googlecloudsdk.api_lib.container.fleet.connectgateway import util
from googlecloudsdk.calliope import base
class GatewayClient:
    """Client for the Connect Gateway API with related helper methods.

  If not provided, the default client is for the GA (v1) track. This client
  is a thin wrapper around the base client, and does not handle any exceptions.

  Fields:
    release_track: The release track of the command [ALPHA, BETA, GA].
    client: The raw GKE Hub API client for the specified release track.
    messages: The matching messages module for the client.
  """

    def __init__(self, release_track: base.ReleaseTrack=util.DEFAULT_TRACK):
        self.release_track = release_track if release_track is not None else util.DEFAULT_TRACK
        self.client = util.GetClientInstance(release_track)
        self.messages = util.GetMessagesModule(release_track)

    def GenerateCredentials(self, name: str, force_use_agent=False, version=None) -> util.TYPES.GenerateCredentialsResponse:
        """Retrieve connection information for accessing a membership through Connect Gateway.

    Args:
      name: The full membership name, in the form
        projects/*/locations/*/memberships/*.
      force_use_agent: Whether to force the use of Connect Agent-based
        transport.
      version: The Connect Gateway version to be used in the resulting
        configuration.

    Returns:
      The GenerateCredentialsResponse message.
    """
        req = self.messages.ConnectgatewayProjectsLocationsMembershipsGenerateCredentialsRequest(name=name, forceUseAgent=force_use_agent, version=version)
        return self.client.projects_locations_memberships.GenerateCredentials(req)