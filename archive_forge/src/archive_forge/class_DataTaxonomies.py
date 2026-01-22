from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.calliope import base
@base.ReleaseTracks(base.ReleaseTrack.ALPHA)
class DataTaxonomies(base.Group):
    """Manage Dataplex Data Taxonomies."""
    category = base.DATA_ANALYTICS_CATEGORY