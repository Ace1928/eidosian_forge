from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.api_lib.app import appengine_api_client as app_engine_api
from googlecloudsdk.api_lib.tasks import GetApiAdapter
from googlecloudsdk.calliope import base as calliope_base
from googlecloudsdk.command_lib.tasks import constants
from googlecloudsdk.core import exceptions
from googlecloudsdk.core import log
def AppEngineAppExists():
    """Returns whether an AppEngine app exists for the current project.

  Previously we were relying on the output of ListLocations for Cloud Tasks &
  Cloud Scheduler to determine if an AppEngine exists. Previous behaviour was
  to return only one location which would be the AppEngine app location and an
  empty list otherwise if no app existed. Now with AppEngine dependency removal,
  ListLocations will return an actual list of valid regions. If an AppEngine app
  does exist, that location will be returned indexed at 0 in the result list.
  Note: We also return False if the user does not have the necessary permissions
  to determine if the project has an AppEngine app or not.

  Returns:
    Boolean representing whether an app exists or not.
  """
    app_engine_api_client = app_engine_api.GetApiClientForTrack(calliope_base.ReleaseTrack.GA)
    try:
        app_engine_api_client.GetApplication()
        found_app = True
    except Exception:
        found_app = False
    return found_app