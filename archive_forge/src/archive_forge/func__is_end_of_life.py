from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.api_lib.container.gkemulticloud import locations as api_util
from googlecloudsdk.command_lib.container.gkemulticloud import constants
def _is_end_of_life(valid_versions, version):
    """Tells if a version is end of life.

  Args:
    valid_versions: A array, contains validVersions are retrieved from
      GetServerConfig (platform dependent).
    version: A string, the GKE version the component is running.

  Returns:
    A boolean value to state if the version on the specified platform is marked
    as end of life.
  """
    for x in valid_versions:
        if x.version == version:
            if x.endOfLife:
                return True
            return False
    return True