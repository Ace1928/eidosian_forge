from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class DatacatalogProjectsLocationsTaxonomiesDeleteRequest(_messages.Message):
    """A DatacatalogProjectsLocationsTaxonomiesDeleteRequest object.

  Fields:
    name: Required. Resource name of the taxonomy to delete. Note: All policy
      tags in this taxonomy are also deleted.
  """
    name = _messages.StringField(1, required=True)