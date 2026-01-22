from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class AiplatformProjectsLocationsNotebookRuntimeTemplatesListRequest(_messages.Message):
    """A AiplatformProjectsLocationsNotebookRuntimeTemplatesListRequest object.

  Fields:
    filter: Optional. An expression for filtering the results of the request.
      For field names both snake_case and camelCase are supported. *
      `notebookRuntimeTemplate` supports = and !=. `notebookRuntimeTemplate`
      represents the NotebookRuntimeTemplate ID, i.e. the last segment of the
      NotebookRuntimeTemplate's resource name. * `display_name` supports = and
      != * `labels` supports general map functions that is: *
      `labels.key=value` - key:value equality * `labels.key:* or labels:key -
      key existence * A key including a space must be quoted. `labels."a
      key"`. * `notebookRuntimeType` supports = and !=. notebookRuntimeType
      enum: [USER_DEFINED, ONE_CLICK]. Some examples: *
      `notebookRuntimeTemplate=notebookRuntimeTemplate123` *
      `displayName="myDisplayName"` * `labels.myKey="myValue"` *
      `notebookRuntimeType=USER_DEFINED`
    orderBy: Optional. A comma-separated list of fields to order by, sorted in
      ascending order. Use "desc" after a field name for descending. Supported
      fields: * `display_name` * `create_time` * `update_time` Example:
      `display_name, create_time desc`.
    pageSize: Optional. The standard list page size.
    pageToken: Optional. The standard list page token. Typically obtained via
      ListNotebookRuntimeTemplatesResponse.next_page_token of the previous
      NotebookService.ListNotebookRuntimeTemplates call.
    parent: Required. The resource name of the Location from which to list the
      NotebookRuntimeTemplates. Format:
      `projects/{project}/locations/{location}`
    readMask: Optional. Mask specifying which fields to read.
  """
    filter = _messages.StringField(1)
    orderBy = _messages.StringField(2)
    pageSize = _messages.IntegerField(3, variant=_messages.Variant.INT32)
    pageToken = _messages.StringField(4)
    parent = _messages.StringField(5, required=True)
    readMask = _messages.StringField(6)