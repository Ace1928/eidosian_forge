from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GoogleCloudDatacatalogV1ImportTaxonomiesResponse(_messages.Message):
    """Response message for ImportTaxonomies.

  Fields:
    taxonomies: Imported taxonomies.
  """
    taxonomies = _messages.MessageField('GoogleCloudDatacatalogV1Taxonomy', 1, repeated=True)