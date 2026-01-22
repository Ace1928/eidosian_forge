from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.api_lib.cloudbuild import transforms
from googlecloudsdk.calliope import base
class Builds(base.Group):
    """Create and manage builds for Google Cloud Build."""
    category = base.CI_CD_CATEGORY

    @staticmethod
    def Args(parser):
        parser.display_info.AddTransforms(transforms.GetTransforms())

    def Filter(self, context, args):
        base.RequireProjectID(args)
        del context, args
        base.EnableUserProjectQuotaWithFallback()