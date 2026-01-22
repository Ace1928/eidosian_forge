from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class DatacatalogProjectsLocationsTaxonomiesImportRequest(_messages.Message):
    """A DatacatalogProjectsLocationsTaxonomiesImportRequest object.

  Fields:
    googleCloudDatacatalogV1ImportTaxonomiesRequest: A
      GoogleCloudDatacatalogV1ImportTaxonomiesRequest resource to be passed as
      the request body.
    parent: Required. Resource name of project that the imported taxonomies
      will belong to.
  """
    googleCloudDatacatalogV1ImportTaxonomiesRequest = _messages.MessageField('GoogleCloudDatacatalogV1ImportTaxonomiesRequest', 1)
    parent = _messages.StringField(2, required=True)