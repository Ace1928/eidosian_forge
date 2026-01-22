from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class DatacatalogProjectsLocationsTaxonomiesGetRequest(_messages.Message):
    """A DatacatalogProjectsLocationsTaxonomiesGetRequest object.

  Fields:
    name: Required. Resource name of the taxonomy to get.
  """
    name = _messages.StringField(1, required=True)