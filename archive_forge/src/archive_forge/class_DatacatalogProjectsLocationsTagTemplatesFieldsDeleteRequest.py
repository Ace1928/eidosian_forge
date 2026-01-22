from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class DatacatalogProjectsLocationsTagTemplatesFieldsDeleteRequest(_messages.Message):
    """A DatacatalogProjectsLocationsTagTemplatesFieldsDeleteRequest object.

  Fields:
    force: Required. If true, deletes this field from any tags that use it.
      Currently, `true` is the only supported value.
    name: Required. The name of the tag template field to delete.
  """
    force = _messages.BooleanField(1)
    name = _messages.StringField(2, required=True)