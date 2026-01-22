from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals
import contextlib
import datetime
import json
from typing import List, MutableSequence, Optional
from apitools.base.py import exceptions as api_exceptions
from googlecloudsdk.api_lib.run import global_methods
from googlecloudsdk.api_lib.run.integrations import api_utils
from googlecloudsdk.api_lib.run.integrations import types_utils
from googlecloudsdk.api_lib.run.integrations import validator
from googlecloudsdk.api_lib.util import apis
from googlecloudsdk.command_lib.run import connection_context
from googlecloudsdk.command_lib.run import flags as run_flags
from googlecloudsdk.command_lib.run import serverless_operations
from googlecloudsdk.command_lib.run.integrations import flags
from googlecloudsdk.command_lib.run.integrations import integration_list_printer
from googlecloudsdk.command_lib.run.integrations import messages_util
from googlecloudsdk.command_lib.run.integrations import stages
from googlecloudsdk.command_lib.run.integrations import typekits_util
from googlecloudsdk.command_lib.run.integrations.typekits import base
from googlecloudsdk.command_lib.runapps import exceptions
from googlecloudsdk.core import properties
from googlecloudsdk.core import resources
from googlecloudsdk.core import yaml
from googlecloudsdk.core.console import progress_tracker
from googlecloudsdk.generated_clients.apis.runapps.v1alpha1 import runapps_v1alpha1_messages
import six
def ListIntegrations(self, integration_type_filter: str, service_name_filter: str, region: str=None, filter_for_type: Optional[str]=None):
    """Returns the list of integrations from the default applications.

    If a '-' is provided for the region, then list applications will be called.
    This is for the global integrations list call.  Any other time
    the default region (either from --region or from gcloud config) will be
    used to fetch the default application.  If the global call is not needed,
    then fetching from a single region will reduce latency and remove the need
    of filtering out non default applications.

    Args:
      integration_type_filter: if populated integration type to filter by.
      service_name_filter: if populated service name to filter by.
      region: GCP region. If not provided, then the region will be pulled from
        the flag or from the config.  Only '-', which is the global region has
        any effect here.
      filter_for_type: the type to filter the list on. if given, the resources
        of that type will not be included in the list, and will only show
        binding to or from that type. if not given, all resources and bindings
        will be shown. for example, for `run integrations list`, it would filter
        for `service`.

    Returns:
      List of Dicts containing name, type, and services.
    """
    endpoint = properties.VALUES.api_endpoint_overrides.runapps.Get()
    if region == ALL_REGIONS and (not _IsLocalHost(endpoint)):
        list_apps = api_utils.ListApplications(self._client, self.ListAppsRequest())
        apps = list_apps.applications if list_apps else []
        apps = _FilterForDefaultApps(apps)
    else:
        app = api_utils.GetApplication(self._client, self.GetAppRef(_DEFAULT_APP_NAME))
        apps = [app] if app else []
    if not apps:
        return []
    output = []
    for app in apps:
        output.extend(self._ParseResourcesForList(app, integration_type_filter, service_name_filter, filter_for_type))
    return output