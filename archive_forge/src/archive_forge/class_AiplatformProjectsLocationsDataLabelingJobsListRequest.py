from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class AiplatformProjectsLocationsDataLabelingJobsListRequest(_messages.Message):
    """A AiplatformProjectsLocationsDataLabelingJobsListRequest object.

  Fields:
    filter: The standard list filter. Supported fields: * `display_name`
      supports `=`, `!=` comparisons, and `:` wildcard. * `state` supports
      `=`, `!=` comparisons. * `create_time` supports `=`, `!=`,`<`, `<=`,`>`,
      `>=` comparisons. `create_time` must be in RFC 3339 format. * `labels`
      supports general map functions that is: `labels.key=value` - key:value
      equality `labels.key:* - key existence Some examples of using the filter
      are: * `state="JOB_STATE_SUCCEEDED" AND display_name:"my_job_*"` *
      `state!="JOB_STATE_FAILED" OR display_name="my_job"` * `NOT
      display_name="my_job"` * `create_time>"2021-05-18T00:00:00Z"` *
      `labels.keyA=valueA` * `labels.keyB:*`
    orderBy: A comma-separated list of fields to order by, sorted in ascending
      order by default. Use `desc` after a field name for descending.
    pageSize: The standard list page size.
    pageToken: The standard list page token.
    parent: Required. The parent of the DataLabelingJob. Format:
      `projects/{project}/locations/{location}`
    readMask: Mask specifying which fields to read. FieldMask represents a set
      of symbolic field paths. For example, the mask can be `paths: "name"`.
      The "name" here is a field in DataLabelingJob. If this field is not set,
      all fields of the DataLabelingJob are returned.
  """
    filter = _messages.StringField(1)
    orderBy = _messages.StringField(2)
    pageSize = _messages.IntegerField(3, variant=_messages.Variant.INT32)
    pageToken = _messages.StringField(4)
    parent = _messages.StringField(5, required=True)
    readMask = _messages.StringField(6)