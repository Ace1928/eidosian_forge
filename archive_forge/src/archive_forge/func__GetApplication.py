from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import abc
from apitools.base.py import encoding
from apitools.base.py import list_pager
from googlecloudsdk.api_lib.app import appengine_api_client
from googlecloudsdk.api_lib.app import operations_util
from googlecloudsdk.api_lib.cloudresourcemanager import projects_api
from googlecloudsdk.api_lib.compute import base_classes
from googlecloudsdk.api_lib.util import apis
from googlecloudsdk.calliope import base
from googlecloudsdk.calliope import exceptions as calliope_exceptions
from googlecloudsdk.command_lib.iam import iam_util
from googlecloudsdk.command_lib.projects import util as projects_util
from googlecloudsdk.core import log
from googlecloudsdk.core import resources
from googlecloudsdk.core import yaml
import six
def _GetApplication(project):
    """Returns the application, given a project."""
    api_client = appengine_api_client.AppengineApiClient.GetApiClient()
    application = resources.REGISTRY.Parse(None, params={'appsId': project}, collection=APPENGINE_APPS_COLLECTION)
    request = api_client.messages.AppengineAppsGetRequest(name=application.RelativeName())
    return api_client.client.apps.Get(request)