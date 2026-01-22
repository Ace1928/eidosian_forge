from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class AiplatformProjectsLocationsModelsListVersionsRequest(_messages.Message):
    """A AiplatformProjectsLocationsModelsListVersionsRequest object.

  Fields:
    filter: An expression for filtering the results of the request. For field
      names both snake_case and camelCase are supported. * `labels` supports
      general map functions that is: * `labels.key=value` - key:value equality
      * `labels.key:* or labels:key - key existence * A key including a space
      must be quoted. `labels."a key"`. Some examples: *
      `labels.myKey="myValue"`
    name: Required. The name of the model to list versions for.
    orderBy: A comma-separated list of fields to order by, sorted in ascending
      order. Use "desc" after a field name for descending. Supported fields: *
      `create_time` * `update_time` Example: `update_time asc, create_time
      desc`.
    pageSize: The standard list page size.
    pageToken: The standard list page token. Typically obtained via
      next_page_token of the previous ListModelVersions call.
    readMask: Mask specifying which fields to read.
  """
    filter = _messages.StringField(1)
    name = _messages.StringField(2, required=True)
    orderBy = _messages.StringField(3)
    pageSize = _messages.IntegerField(4, variant=_messages.Variant.INT32)
    pageToken = _messages.StringField(5)
    readMask = _messages.StringField(6)