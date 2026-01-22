from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class DatacatalogProjectsLocationsTaxonomiesExportRequest(_messages.Message):
    """A DatacatalogProjectsLocationsTaxonomiesExportRequest object.

  Fields:
    parent: Required. Resource name of the project that the exported
      taxonomies belong to.
    serializedTaxonomies: Serialized export taxonomies that contain all the
      policy tags as nested protocol buffers.
    taxonomies: Required. Resource names of the taxonomies to export.
  """
    parent = _messages.StringField(1, required=True)
    serializedTaxonomies = _messages.BooleanField(2)
    taxonomies = _messages.StringField(3, repeated=True)