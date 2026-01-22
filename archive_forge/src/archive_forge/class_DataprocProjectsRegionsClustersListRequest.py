from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class DataprocProjectsRegionsClustersListRequest(_messages.Message):
    """A DataprocProjectsRegionsClustersListRequest object.

  Fields:
    filter: Optional. A filter constraining the clusters to list. Filters are
      case-sensitive and have the following syntax:field = value AND field =
      value ...where field is one of status.state, clusterName, or
      labels.[KEY], and [KEY] is a label key. value can be * to match all
      values. status.state can be one of the following: ACTIVE, INACTIVE,
      CREATING, RUNNING, ERROR, DELETING, UPDATING, STOPPING, or STOPPED.
      ACTIVE contains the CREATING, UPDATING, and RUNNING states. INACTIVE
      contains the DELETING, ERROR, STOPPING, and STOPPED states. clusterName
      is the name of the cluster provided at creation time. Only the logical
      AND operator is supported; space-separated items are treated as having
      an implicit AND operator.Example filter:status.state = ACTIVE AND
      clusterName = mycluster AND labels.env = staging AND labels.starred = *
    pageSize: Optional. The standard List page size.
    pageToken: Optional. The standard List page token.
    projectId: Required. The ID of the Google Cloud Platform project that the
      cluster belongs to.
    region: Required. The Dataproc region in which to handle the request.
  """
    filter = _messages.StringField(1)
    pageSize = _messages.IntegerField(2, variant=_messages.Variant.INT32)
    pageToken = _messages.StringField(3)
    projectId = _messages.StringField(4, required=True)
    region = _messages.StringField(5, required=True)