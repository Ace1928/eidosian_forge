from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class DatacatalogProjectsTaxonomiesCategoriesGetRequest(_messages.Message):
    """A DatacatalogProjectsTaxonomiesCategoriesGetRequest object.

  Fields:
    name: Required. Resource name of the category to be returned.
  """
    name = _messages.StringField(1, required=True)