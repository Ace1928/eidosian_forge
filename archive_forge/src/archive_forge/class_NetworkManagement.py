from googlecloudsdk.calliope import base
@base.ReleaseTracks(base.ReleaseTrack.GA, base.ReleaseTrack.BETA, base.ReleaseTrack.ALPHA)
class NetworkManagement(base.Group):
    """Manage Network Management resources."""
    category = base.NETWORKING_CATEGORY

    def Filter(self, context, args):
        base.RequireProjectID(args)
        del context, args