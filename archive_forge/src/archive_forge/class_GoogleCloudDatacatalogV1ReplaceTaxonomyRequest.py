from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GoogleCloudDatacatalogV1ReplaceTaxonomyRequest(_messages.Message):
    """Request message for ReplaceTaxonomy.

  Fields:
    serializedTaxonomy: Required. Taxonomy to update along with its child
      policy tags.
  """
    serializedTaxonomy = _messages.MessageField('GoogleCloudDatacatalogV1SerializedTaxonomy', 1)