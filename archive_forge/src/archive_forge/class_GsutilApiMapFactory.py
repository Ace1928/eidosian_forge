from __future__ import absolute_import
from __future__ import print_function
from __future__ import division
from __future__ import unicode_literals
from gslib.boto_translation import BotoTranslation
from gslib.gcs_json_api import GcsJsonApi
class GsutilApiMapFactory(object):
    """Factory the generates the default gsutil API map.

    The API map determines which Cloud API implementation is used for a given
    command.  A valid API map is defined as:
    {
      (key) ApiMapConstants.API_MAP : (value) Gsutil API class map (as
          described in GsutilApiClassMapFactory comments).
      (key) ApiMapConstants.SUPPORT_MAP : (value) {
        (key) Provider prefix used in URI strings.
        (value) list of ApiSelectors supported by the command for this provider.
      }
      (key) ApiMapConstants.DEFAULT_MAP : (value) {
        (key) Provider prefix used in URI strings.
        (value) Default ApiSelector for this command and provider.
      }
    }
  """

    @classmethod
    def GetApiMap(cls, gsutil_api_class_map_factory, support_map, default_map):
        """Creates a GsutilApiMap for use by the command from the inputs.

    Args:
      gsutil_api_class_map_factory: Factory defining a GetClassMap() function
                                    adhering to GsutilApiClassMapFactory
                                    semantics.
      support_map: Entries for ApiMapConstants.SUPPORT_MAP as described above.
      default_map: Entries for ApiMapConstants.DEFAULT_MAP as described above.

    Returns:
      GsutilApiMap generated from the inputs.
    """
        return {ApiMapConstants.API_MAP: gsutil_api_class_map_factory.GetClassMap(), ApiMapConstants.SUPPORT_MAP: support_map, ApiMapConstants.DEFAULT_MAP: default_map}