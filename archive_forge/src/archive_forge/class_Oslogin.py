from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.calliope import base
class Oslogin(base.Group):
    """Create and manipulate Compute Engine OS Login resources."""
    category = base.TOOLS_CATEGORY
    detailed_help = DETAILED_HELP

    def Filter(self, context, args):
        del context, args
        base.EnableUserProjectQuotaWithFallback()