from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class DataprocProjectsRegionsJobsListRequest(_messages.Message):
    """A DataprocProjectsRegionsJobsListRequest object.

  Enums:
    JobStateMatcherValueValuesEnum: Optional. Specifies enumerated categories
      of jobs to list. (default = match ALL jobs).If filter is provided,
      jobStateMatcher will be ignored.

  Fields:
    clusterName: Optional. If set, the returned jobs list includes only jobs
      that were submitted to the named cluster.
    filter: Optional. A filter constraining the jobs to list. Filters are
      case-sensitive and have the following syntax:field = value AND field =
      value ...where field is status.state or labels.[KEY], and [KEY] is a
      label key. value can be * to match all values. status.state can be
      either ACTIVE or NON_ACTIVE. Only the logical AND operator is supported;
      space-separated items are treated as having an implicit AND
      operator.Example filter:status.state = ACTIVE AND labels.env = staging
      AND labels.starred = *
    jobStateMatcher: Optional. Specifies enumerated categories of jobs to
      list. (default = match ALL jobs).If filter is provided, jobStateMatcher
      will be ignored.
    pageSize: Optional. The number of results to return in each response.
    pageToken: Optional. The page token, returned by a previous call, to
      request the next page of results.
    projectId: Required. The ID of the Google Cloud Platform project that the
      job belongs to.
    region: Required. The Dataproc region in which to handle the request.
  """

    class JobStateMatcherValueValuesEnum(_messages.Enum):
        """Optional. Specifies enumerated categories of jobs to list. (default =
    match ALL jobs).If filter is provided, jobStateMatcher will be ignored.

    Values:
      ALL: Match all jobs, regardless of state.
      ACTIVE: Only match jobs in non-terminal states: PENDING, RUNNING, or
        CANCEL_PENDING.
      NON_ACTIVE: Only match jobs in terminal states: CANCELLED, DONE, or
        ERROR.
    """
        ALL = 0
        ACTIVE = 1
        NON_ACTIVE = 2
    clusterName = _messages.StringField(1)
    filter = _messages.StringField(2)
    jobStateMatcher = _messages.EnumField('JobStateMatcherValueValuesEnum', 3)
    pageSize = _messages.IntegerField(4, variant=_messages.Variant.INT32)
    pageToken = _messages.StringField(5)
    projectId = _messages.StringField(6, required=True)
    region = _messages.StringField(7, required=True)