from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.api_lib.container.gkemulticloud import locations as api_util
from googlecloudsdk.command_lib.container.gkemulticloud import constants
def _load_valid_versions(platform, location_ref):
    """Loads the valid version in respect to the platform via server config.

  Args:
    platform: A string, the platform the component is on {AWS,Azure}.
    location_ref:  A resource object, the pathing portion the url, used to get
      the proper server config.

  Returns:
    Returns the list of valid version that were obtained in the getServerConfig
    call.
  """
    client = api_util.LocationsClient()
    if platform == constants.AZURE:
        return client.GetAzureServerConfig(location_ref).validVersions
    elif platform == constants.AWS:
        return client.GetAwsServerConfig(location_ref).validVersions
    else:
        return None