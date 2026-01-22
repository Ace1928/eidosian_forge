from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class AiplatformProjectsLocationsPersistentResourcesListRequest(_messages.Message):
    """A AiplatformProjectsLocationsPersistentResourcesListRequest object.

  Fields:
    pageSize: Optional. The standard list page size.
    pageToken: Optional. The standard list page token. Typically obtained via
      ListPersistentResourceResponse.next_page_token of the previous
      PersistentResourceService.ListPersistentResource call.
    parent: Required. The resource name of the Location to list the
      PersistentResources from. Format:
      `projects/{project}/locations/{location}`
  """
    pageSize = _messages.IntegerField(1, variant=_messages.Variant.INT32)
    pageToken = _messages.StringField(2)
    parent = _messages.StringField(3, required=True)