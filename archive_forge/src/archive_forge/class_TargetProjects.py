from googlecloudsdk.calliope import base
@base.ReleaseTracks(base.ReleaseTrack.ALPHA)
class TargetProjects(base.Group):
    """Manage Target Projects.

  target-projects sub-group is used to manage Target Project resources of the
  Migrate to Virtual Machines service.
  Target projects are defined for each customer project in the global location.
  A Target Project could be used as the target project of various migration
  commands.
  VM Migration API must be enabled in your project.

  ## List Target Projects
  gcloud alpha migration vms target-projects list
  """