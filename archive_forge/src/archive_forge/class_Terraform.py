from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.calliope import base
@base.ReleaseTracks(base.ReleaseTrack.ALPHA, base.ReleaseTrack.BETA)
class Terraform(base.Group):
    """The command group for Terraform provider configuration."""
    category = base.DECLARATIVE_CONFIGURATION_CATEGORY

    def Filter(self, context, args):
        base.RequireProjectID(args)
        del context, args