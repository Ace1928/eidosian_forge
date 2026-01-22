from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.calliope import base
@base.ReleaseTracks(base.ReleaseTrack.ALPHA)
class StacksTypes(base.Group):
    """View available Stacks Types.

  This set of commands can be used to view Stacks Types.
  """
    detailed_help = {'EXAMPLES': '\n          To list available Stacks types, run:\n\n            $ {command} list\n      '}