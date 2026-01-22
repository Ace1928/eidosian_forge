from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.calliope import base
class SQL(base.Group):
    """Create and manage Google Cloud SQL databases."""
    category = base.DATABASES_CATEGORY
    detailed_help = DETAILED_HELP

    def Filter(self, context, args):
        del context, args
        base.EnableUserProjectQuotaWithFallback()