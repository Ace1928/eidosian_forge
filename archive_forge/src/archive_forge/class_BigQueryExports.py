from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.calliope import base
@base.ReleaseTracks(base.ReleaseTrack.GA)
class BigQueryExports(base.Group):
    """Manage Cloud SCC (Security Command Center) BigQuery exports."""
    category = base.SECURITY_CATEGORY