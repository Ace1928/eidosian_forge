from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GoogleCloudDatacatalogV1ExportTaxonomiesResponse(_messages.Message):
    """Response message for ExportTaxonomies.

  Fields:
    taxonomies: List of taxonomies and policy tags as nested protocol buffers.
  """
    taxonomies = _messages.MessageField('GoogleCloudDatacatalogV1SerializedTaxonomy', 1, repeated=True)