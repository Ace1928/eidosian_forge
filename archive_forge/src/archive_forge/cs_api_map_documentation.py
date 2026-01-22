from __future__ import absolute_import
from __future__ import print_function
from __future__ import division
from __future__ import unicode_literals
from gslib.boto_translation import BotoTranslation
from gslib.gcs_json_api import GcsJsonApi
Creates a GsutilApiMap for use by the command from the inputs.

    Args:
      gsutil_api_class_map_factory: Factory defining a GetClassMap() function
                                    adhering to GsutilApiClassMapFactory
                                    semantics.
      support_map: Entries for ApiMapConstants.SUPPORT_MAP as described above.
      default_map: Entries for ApiMapConstants.DEFAULT_MAP as described above.

    Returns:
      GsutilApiMap generated from the inputs.
    