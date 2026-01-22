from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.calliope import base
class Anthos(base.Group):
    """Anthos command Group."""
    category = base.ANTHOS_CLI_CATEGORY

    def Filter(self, context, args):
        base.RequireProjectID(args)
        del context, args