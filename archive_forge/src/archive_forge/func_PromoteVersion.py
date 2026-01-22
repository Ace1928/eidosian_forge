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
def PromoteVersion(all_services, new_version, api_client, stop_previous_version, wait_for_stop_version):
    """Promote the new version to receive all traffic.

  First starts the new version if it is not running.

  Additionally, stops the previous version if stop_previous_version is True and
  it is possible to stop the previous version.

  Args:
    all_services: {str, Service}, A mapping of service id to Service objects
      for all services in the app.
    new_version: Version, The version to promote.
    api_client: appengine_api_client.AppengineApiClient to use to make requests.
    stop_previous_version: bool, True to stop the previous version which was
      receiving all traffic, if any.
    wait_for_stop_version: bool, indicating whether to wait for stop operation
    to finish.
  """
    old_default_version = None
    if stop_previous_version:
        old_default_version = _GetPreviousVersion(all_services, new_version, api_client)
    new_version_resource = new_version.GetVersionResource(api_client)
    status_enum = api_client.messages.Version.ServingStatusValueValuesEnum
    if new_version_resource and new_version_resource.servingStatus == status_enum.STOPPED:
        log.status.Print('Starting version [{0}] before promoting it.'.format(new_version))
        api_client.StartVersion(new_version.service, new_version.id, block=True)
    _SetDefaultVersion(new_version, api_client)
    if old_default_version:
        _StopPreviousVersionIfApplies(old_default_version, api_client, wait_for_stop_version)