from googlecloudsdk.calliope import base
@base.ReleaseTracks(base.ReleaseTrack.ALPHA)
class Vms(base.Group):
    """Provides Migrate to Virtual Machines (VM migration) service functionality.

  The gcloud alpha migration vms command group provides the CLI for
  the Migrate to Virtual Machines API.
  Google Cloud Migrate to Virtual Machines migrates VMs from on-premises data
  center and other cloud providers to Google Compute Engine virtual machine (VM)
  instances.
  VM Migration API must be enabled in your project.
  """