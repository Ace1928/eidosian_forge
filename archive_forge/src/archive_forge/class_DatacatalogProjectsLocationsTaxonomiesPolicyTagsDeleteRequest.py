from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class DatacatalogProjectsLocationsTaxonomiesPolicyTagsDeleteRequest(_messages.Message):
    """A DatacatalogProjectsLocationsTaxonomiesPolicyTagsDeleteRequest object.

  Fields:
    name: Required. Resource name of the policy tag to delete. Note: All of
      its descendant policy tags are also deleted.
  """
    name = _messages.StringField(1, required=True)