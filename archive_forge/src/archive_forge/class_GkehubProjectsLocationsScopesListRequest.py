from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GkehubProjectsLocationsScopesListRequest(_messages.Message):
    """A GkehubProjectsLocationsScopesListRequest object.

  Fields:
    pageSize: Optional. When requesting a 'page' of resources, `page_size`
      specifies number of resources to return. If unspecified or set to 0, all
      resources will be returned.
    pageToken: Optional. Token returned by previous call to `ListScopes` which
      specifies the position in the list from where to continue listing the
      resources.
    parent: Required. The parent (project and location) where the Scope will
      be listed. Specified in the format `projects/*/locations/*`.
  """
    pageSize = _messages.IntegerField(1, variant=_messages.Variant.INT32)
    pageToken = _messages.StringField(2)
    parent = _messages.StringField(3, required=True)