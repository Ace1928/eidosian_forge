from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class MediaassetProjectsLocationsCatalogsPatchRequest(_messages.Message):
    """A MediaassetProjectsLocationsCatalogsPatchRequest object.

  Fields:
    catalog: A Catalog resource to be passed as the request body.
    name: The resource name of the catalog, in the following form:
      `projects/{project}/locations/{location}/catalogs/{catalog}`.
    updateMask: update_mask is a comma separated fields used to specify
      changes in catalog. The fields specified in the update_mask are relative
      to the resource, not the full request. A field will be overwritten if it
      is in the mask. If the user does not provide a mask then all fields will
      be overwritten.
  """
    catalog = _messages.MessageField('Catalog', 1)
    name = _messages.StringField(2, required=True)
    updateMask = _messages.StringField(3)