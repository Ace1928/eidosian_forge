from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GoogleCloudDatacatalogV1beta1ListTaxonomiesResponse(_messages.Message):
    """Response message for ListTaxonomies.

  Fields:
    nextPageToken: Token used to retrieve the next page of results, or empty
      if there are no more results in the list.
    taxonomies: Taxonomies that the project contains.
  """
    nextPageToken = _messages.StringField(1)
    taxonomies = _messages.MessageField('GoogleCloudDatacatalogV1beta1Taxonomy', 2, repeated=True)