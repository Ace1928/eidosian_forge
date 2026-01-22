from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.calliope import base
@base.ReleaseTracks(base.ReleaseTrack.GA, base.ReleaseTrack.BETA)
class Versions(base.Group):
    """View and manage your App Engine versions.

  This set of commands can be used to view and manage your existing App Engine
  versions.

  To create new deployments, use `{parent_command} deploy`.

  For more information on App Engine versions, see:
  https://cloud.google.com/appengine/docs/python/an-overview-of-app-engine

  ## EXAMPLES

  To list your deployed versions, run:

    $ {command} list
  """
    category = base.APP_ENGINE_CATEGORY