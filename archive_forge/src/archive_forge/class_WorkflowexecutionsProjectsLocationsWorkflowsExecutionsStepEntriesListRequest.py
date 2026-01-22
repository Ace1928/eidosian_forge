from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
class WorkflowexecutionsProjectsLocationsWorkflowsExecutionsStepEntriesListRequest(_messages.Message):
    """A
  WorkflowexecutionsProjectsLocationsWorkflowsExecutionsStepEntriesListRequest
  object.

  Fields:
    filter: Optional. Filters applied to the `[StepEntries.ListStepEntries]`
      results. The following fields are supported for filtering: `entryId`,
      `createTime`, `updateTime`, `routine`, `step`, `stepType`, `state`. For
      details, see AIP-160. For example, if you are using the Google APIs
      Explorer: `state="SUCCEEDED"` or `createTime>"2023-08-01" AND
      state="FAILED"`
    orderBy: Optional. Comma-separated list of fields that specify the
      ordering applied to the `[StepEntries.ListStepEntries]` results. By
      default the ordering is based on ascending `entryId`. The following
      fields are supported for ordering: `entryId`, `createTime`,
      `updateTime`, `routine`, `step`, `stepType`, `state`. For details, see
      AIP-132.
    pageSize: Optional. Number of step entries to return per call. The default
      max is 1000.
    pageToken: Optional. A page token, received from a previous
      `ListStepEntries` call. Provide this to retrieve the subsequent page.
      When paginating, all other parameters provided to `ListStepEntries` must
      match the call that provided the page token.
    parent: Required. Name of the workflow execution to list entries for.
      Format: projects/{project}/locations/{location}/workflows/{workflow}/exe
      cutions/{execution}/stepEntries/
    skip: Optional. The number of step entries to skip. It can be used with or
      without a pageToken. If used with a pageToken, then it indicates the
      number of step entries to skip starting from the requested page.
  """
    filter = _messages.StringField(1)
    orderBy = _messages.StringField(2)
    pageSize = _messages.IntegerField(3, variant=_messages.Variant.INT32)
    pageToken = _messages.StringField(4)
    parent = _messages.StringField(5, required=True)
    skip = _messages.IntegerField(6, variant=_messages.Variant.INT32)