from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals
import re
from typing import List, Optional
from apitools.base.py import encoding as apitools_encoding
from apitools.base.py import exceptions as apitools_exceptions
from googlecloudsdk.api_lib.util import apis
from googlecloudsdk.api_lib.util import exceptions as api_lib_exceptions
from googlecloudsdk.api_lib.util import waiter
from googlecloudsdk.command_lib.runapps import exceptions
from googlecloudsdk.core import log
from googlecloudsdk.core import resources
from googlecloudsdk.core.util import encoding
from googlecloudsdk.core.util import retry
from googlecloudsdk.generated_clients.apis.runapps.v1alpha1 import runapps_v1alpha1_client
from googlecloudsdk.generated_clients.apis.runapps.v1alpha1 import runapps_v1alpha1_messages
def _HandleLocationError(region: str, error: Exception) -> Exception:
    """Get the metadata message for the deployment operation.

  Args:
    region: target region of the request.
    error: original HttpError.

  Raises:
    UnsupportedIntegrationsLocationError if it's location error. Otherwise
    raise the original error.
  """
    parsed_err = api_lib_exceptions.HttpException(error)
    if _LOCATION_ERROR_REGEX.match(parsed_err.payload.status_message):
        raise exceptions.UnsupportedIntegrationsLocationError('Location {} is not found or access is unauthorized.'.format(region))
    raise error