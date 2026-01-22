from googlecloudsdk.calliope import base
@base.ReleaseTracks(base.ReleaseTrack.ALPHA, base.ReleaseTrack.BETA)
class UserWorkloadsConfigMaps(base.Group):
    """Create and manage user workloads ConfigMaps of environment.

  The {command} command group manages user workloads ConfigMaps of Cloud
  Composer
  environments.
  """