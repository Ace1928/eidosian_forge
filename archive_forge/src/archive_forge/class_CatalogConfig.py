from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class CatalogConfig(_messages.Message):
    """Configures the catalog of assets.

  Fields:
    catalog: Required. Reference to a catalog to populate with assets:
      `projects/{project}/locations/{location}/catalogs/{name}`.
  """
    catalog = _messages.StringField(1)