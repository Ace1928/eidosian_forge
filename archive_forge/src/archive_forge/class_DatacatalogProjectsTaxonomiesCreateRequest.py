from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class DatacatalogProjectsTaxonomiesCreateRequest(_messages.Message):
    """A DatacatalogProjectsTaxonomiesCreateRequest object.

  Fields:
    googleCloudDatacatalogV1alpha3Taxonomy: A
      GoogleCloudDatacatalogV1alpha3Taxonomy resource to be passed as the
      request body.
    parent: Required. Resource name of the project that the newly created
      taxonomy belongs to.
  """
    googleCloudDatacatalogV1alpha3Taxonomy = _messages.MessageField('GoogleCloudDatacatalogV1alpha3Taxonomy', 1)
    parent = _messages.StringField(2, required=True)