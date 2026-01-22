from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.calliope import base
@base.ReleaseTracks(base.ReleaseTrack.ALPHA, base.ReleaseTrack.GA)
class EdgeCache(base.Group):
    """Manage Media CDN resources."""
    category = base.NETWORKING_CATEGORY
    detailed_help = {'DESCRIPTION': 'Manage Media CDN resources.', 'EXAMPLES': '\n          To list EdgeCacheService resources in the active Cloud Platform\n          project, run:\n\n            $ {command} services list\n\n          To create an EdgeCacheOrigin resource named \'my-origin\' that\n          points to a Cloud Storage bucket, run:\n\n            $ {command} origins create my-origin --origin-address="gs://bucket"\n\n          To import an EdgeCacheService resource configuration from a YAML\n          definition, run:\n\n            $ {command} services import my-service --source=config.yaml\n\n          To describe an EdgeCacheKeyset resource named \'my-keyset\', run:\n\n            $ {command} keysets describe my-keyset\n          '}

    def Filter(self, context, args):
        base.RequireProjectID(args)
        del context, args