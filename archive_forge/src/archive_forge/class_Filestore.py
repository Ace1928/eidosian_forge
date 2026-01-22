from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.calliope import base
class Filestore(base.Group):
    """Create and manipulate Filestore resources."""
    detailed_help = DETAILED_HELP
    category = base.STORAGE_CATEGORY

    def Filter(self, context, args):
        base.RequireProjectID(args)
        del context, args