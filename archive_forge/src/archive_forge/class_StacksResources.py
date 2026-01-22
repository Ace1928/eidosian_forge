from googlecloudsdk.calliope import base
@base.ReleaseTracks(base.ReleaseTrack.ALPHA)
class StacksResources(base.Group):
    """Stacks resources command group.

  This set of commands can be used to view Cloud Run resources.
  """
    detailed_help = {'EXAMPLES': '\n          To list Stacks resources, run:\n\n            $ {command} list\n      '}