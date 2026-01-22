from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import re
from apitools.base.py import exceptions as apitools_exceptions
from googlecloudsdk.api_lib.app import env
from googlecloudsdk.api_lib.app import metric_names
from googlecloudsdk.api_lib.app import operations_util
from googlecloudsdk.core import exceptions
from googlecloudsdk.core import log
from googlecloudsdk.core import metrics
from googlecloudsdk.core.util import retry
from googlecloudsdk.core.util import text
from googlecloudsdk.core.util import times
import six
from six.moves import map  # pylint: disable=redefined-builtin
def _GetPreviousVersion(all_services, new_version, api_client):
    """Get the previous default version of which new_version is replacing.

  If there is no such version, return None.

  Args:
    all_services: {str, Service}, A mapping of service id to Service objects
      for all services in the app.
    new_version: Version, The version to promote.
    api_client: appengine_api_client.AppengineApiClient, The client for talking
      to the App Engine Admin API.

  Returns:
    Version, The previous version or None.
  """
    service = all_services.get(new_version.service, None)
    if not service:
        return None
    for old_version in api_client.ListVersions([service]):
        if old_version.IsReceivingAllTraffic() and old_version.id != new_version.id:
            return old_version