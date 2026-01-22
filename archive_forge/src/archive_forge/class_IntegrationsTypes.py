from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.calliope import base
from googlecloudsdk.command_lib.run import flags
from googlecloudsdk.command_lib.run import platforms
@base.ReleaseTracks(base.ReleaseTrack.ALPHA, base.ReleaseTrack.BETA)
class IntegrationsTypes(base.Group):
    """View available Cloud Run (fully managed) integrations types.

  This set of commands can be used to view Cloud Run
  integrations types.
  """
    detailed_help = {'EXAMPLES': '\n          To list available integrations types, run:\n\n            $ {command} list\n      '}