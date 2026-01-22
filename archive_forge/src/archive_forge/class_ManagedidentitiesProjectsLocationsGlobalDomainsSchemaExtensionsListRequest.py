from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class ManagedidentitiesProjectsLocationsGlobalDomainsSchemaExtensionsListRequest(_messages.Message):
    """A
  ManagedidentitiesProjectsLocationsGlobalDomainsSchemaExtensionsListRequest
  object.

  Fields:
    filter: Optional. Filter specifying constraints of a list operation. For
      example, `SchemaExtension.name="projects/proj-
      test/locations/global/domains/test.com/schemaExtensions/s-123"`.
    orderBy: Optional. Specifies the ordering of results following syntax at
      https://cloud.google.com/apis/design/design_patterns#sorting_order.
    pageSize: Optional. The maximum number of items to return. The maximum
      value is 1000; values above 1000 will be coerced to 1000. If not
      specified, a default value of 1000 will be used by the service.
      Regardless of the page_size value, the response may include a partial
      list and a caller should only rely on response. next_page_token to
      determine if there are more instances left to be queried.
    pageToken: Optional. The next_page_token value returned from a previous
      List request, if any.
    parent: Required. The domain resource name using the form:
      `projects/{project_id}/locations/global/domains/{domain_name}`
  """
    filter = _messages.StringField(1)
    orderBy = _messages.StringField(2)
    pageSize = _messages.IntegerField(3, variant=_messages.Variant.INT32)
    pageToken = _messages.StringField(4)
    parent = _messages.StringField(5, required=True)