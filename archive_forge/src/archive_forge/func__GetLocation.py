from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import os
from apitools.base.py import exceptions as apitools_exceptions
from apitools.base.py import list_pager
from googlecloudsdk.api_lib.util import apis
from googlecloudsdk.calliope import arg_parsers
from googlecloudsdk.command_lib.tasks import app
from googlecloudsdk.command_lib.tasks import constants
from googlecloudsdk.core import exceptions
from googlecloudsdk.core import log
from googlecloudsdk.core import properties
from googlecloudsdk.core import resources
from googlecloudsdk.core.util import encoding
from googlecloudsdk.core.util import http_encoding
def _GetLocation(self, project):
    """Gets the location from the Cloud Scheduler API."""
    try:
        client = _GetSchedulerClient()
        messages = _GetSchedulerMessages()
        request = messages.CloudschedulerProjectsLocationsListRequest(name='projects/{}'.format(project))
        locations = list(list_pager.YieldFromList(client.projects_locations, request, batch_size=2, limit=2, field='locations', batch_size_attribute='pageSize'))
        if len(locations) >= 1:
            location = locations[0].labels.additionalProperties[0].value
            if len(locations) > 1 and (not _DoesCommandRequireAppEngineApp()):
                log.warning(constants.APP_ENGINE_DEFAULT_LOCATION_WARNING.format(location))
            return location
        return None
    except apitools_exceptions.HttpNotFoundError:
        return None