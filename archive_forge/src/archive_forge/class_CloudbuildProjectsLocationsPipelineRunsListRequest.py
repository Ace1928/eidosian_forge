from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class CloudbuildProjectsLocationsPipelineRunsListRequest(_messages.Message):
    """A CloudbuildProjectsLocationsPipelineRunsListRequest object.

  Fields:
    filter: Filter for the results.
    pageSize: Number of results to return in the list.
    pageToken: Page start.
    parent: Required. The parent, which owns this collection of PipelineRuns.
      Format: projects/{project}/locations/{location}
  """
    filter = _messages.StringField(1)
    pageSize = _messages.IntegerField(2, variant=_messages.Variant.INT32)
    pageToken = _messages.StringField(3)
    parent = _messages.StringField(4, required=True)