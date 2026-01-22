from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class DataplexProjectsLocationsDataTaxonomiesGetRequest(_messages.Message):
    """A DataplexProjectsLocationsDataTaxonomiesGetRequest object.

  Fields:
    name: Required. The resource name of the DataTaxonomy: projects/{project_n
      umber}/locations/{location_id}/dataTaxonomies/{data_taxonomy_id}
  """
    name = _messages.StringField(1, required=True)