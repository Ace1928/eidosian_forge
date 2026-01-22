from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class AiplatformProjectsLocationsPipelineJobsListRequest(_messages.Message):
    """A AiplatformProjectsLocationsPipelineJobsListRequest object.

  Fields:
    filter: Lists the PipelineJobs that match the filter expression. The
      following fields are supported: * `pipeline_name`: Supports `=` and `!=`
      comparisons. * `display_name`: Supports `=`, `!=` comparisons, and `:`
      wildcard. * `pipeline_job_user_id`: Supports `=`, `!=` comparisons, and
      `:` wildcard. for example, can check if pipeline's display_name contains
      *step* by doing display_name:\\"*step*\\" * `state`: Supports `=` and `!=`
      comparisons. * `create_time`: Supports `=`, `!=`, `<`, `>`, `<=`, and
      `>=` comparisons. Values must be in RFC 3339 format. * `update_time`:
      Supports `=`, `!=`, `<`, `>`, `<=`, and `>=` comparisons. Values must be
      in RFC 3339 format. * `end_time`: Supports `=`, `!=`, `<`, `>`, `<=`,
      and `>=` comparisons. Values must be in RFC 3339 format. * `labels`:
      Supports key-value equality and key presence. * `template_uri`: Supports
      `=`, `!=` comparisons, and `:` wildcard. * `template_metadata.version`:
      Supports `=`, `!=` comparisons, and `:` wildcard. Filter expressions can
      be combined together using logical operators (`AND` & `OR`). For
      example: `pipeline_name="test" AND create_time>"2020-05-18T13:30:00Z"`.
      The syntax to define filter expression is based on
      https://google.aip.dev/160. Examples: *
      `create_time>"2021-05-18T00:00:00Z" OR
      update_time>"2020-05-18T00:00:00Z"` PipelineJobs created or updated
      after 2020-05-18 00:00:00 UTC. * `labels.env = "prod"` PipelineJobs with
      label "env" set to "prod".
    orderBy: A comma-separated list of fields to order by. The default sort
      order is in ascending order. Use "desc" after a field name for
      descending. You can have multiple order_by fields provided e.g.
      "create_time desc, end_time", "end_time, start_time, update_time" For
      example, using "create_time desc, end_time" will order results by create
      time in descending order, and if there are multiple jobs having the same
      create time, order them by the end time in ascending order. if order_by
      is not specified, it will order by default order is create time in
      descending order. Supported fields: * `create_time` * `update_time` *
      `end_time` * `start_time`
    pageSize: The standard list page size.
    pageToken: The standard list page token. Typically obtained via
      ListPipelineJobsResponse.next_page_token of the previous
      PipelineService.ListPipelineJobs call.
    parent: Required. The resource name of the Location to list the
      PipelineJobs from. Format: `projects/{project}/locations/{location}`
    readMask: Mask specifying which fields to read.
  """
    filter = _messages.StringField(1)
    orderBy = _messages.StringField(2)
    pageSize = _messages.IntegerField(3, variant=_messages.Variant.INT32)
    pageToken = _messages.StringField(4)
    parent = _messages.StringField(5, required=True)
    readMask = _messages.StringField(6)