from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import copy
import json
import operator
from apitools.base.py import encoding
from apitools.base.py import exceptions as apitools_exceptions
from apitools.base.py import list_pager
from googlecloudsdk.api_lib.app import build as app_cloud_build
from googlecloudsdk.api_lib.app import env
from googlecloudsdk.api_lib.app import exceptions
from googlecloudsdk.api_lib.app import instances_util
from googlecloudsdk.api_lib.app import operations_util
from googlecloudsdk.api_lib.app import region_util
from googlecloudsdk.api_lib.app import service_util
from googlecloudsdk.api_lib.app import util
from googlecloudsdk.api_lib.app import version_util
from googlecloudsdk.api_lib.app.api import appengine_api_client_base
from googlecloudsdk.api_lib.cloudbuild import logs as cloudbuild_logs
from googlecloudsdk.calliope import base as calliope_base
from googlecloudsdk.core import log
from googlecloudsdk.core import properties
from googlecloudsdk.core import resources
from googlecloudsdk.third_party.appengine.admin.tools.conversion import convert_yaml
import six
from six.moves import filter  # pylint: disable=redefined-builtin
from six.moves import map  # pylint: disable=redefined-builtin
def GetAllInstances(self, service=None, version=None, version_filter=None):
    """Generator of all instances, optionally filtering by service or version.

    Args:
      service: str, the ID of the service to filter by.
      version: str, the ID of the version to filter by.
      version_filter: filter function accepting version_util.Version

    Returns:
      generator of instance_util.Instance
    """
    services = self.ListServices()
    log.debug('All services: {0}'.format(services))
    services = service_util.GetMatchingServices(services, [service] if service else None)
    versions = self.ListVersions(services)
    log.debug('Versions: {0}'.format(list(map(str, versions))))
    versions = version_util.GetMatchingVersions(versions, [version] if version else None, service)
    versions = list(filter(version_filter, versions))
    return self.ListInstances(versions)