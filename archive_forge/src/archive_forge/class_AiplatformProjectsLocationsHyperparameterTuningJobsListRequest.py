from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class AiplatformProjectsLocationsHyperparameterTuningJobsListRequest(_messages.Message):
    """A AiplatformProjectsLocationsHyperparameterTuningJobsListRequest object.

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
    pageSize: The standard list page size.
    pageToken: The standard list page token. Typically obtained via
      ListHyperparameterTuningJobsResponse.next_page_token of the previous
      JobService.ListHyperparameterTuningJobs call.
    parent: Required. The resource name of the Location to list the
      HyperparameterTuningJobs from. Format:
      `projects/{project}/locations/{location}`
    readMask: Mask specifying which fields to read.
  """
    filter = _messages.StringField(1)
    pageSize = _messages.IntegerField(2, variant=_messages.Variant.INT32)
    pageToken = _messages.StringField(3)
    parent = _messages.StringField(4, required=True)
    readMask = _messages.StringField(5)