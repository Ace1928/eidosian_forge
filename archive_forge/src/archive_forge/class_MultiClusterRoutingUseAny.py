from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class MultiClusterRoutingUseAny(_messages.Message):
    """Read/write requests are routed to the nearest cluster in the instance,
  and will fail over to the nearest cluster that is available in the event of
  transient errors or delays. Clusters in a region are considered equidistant.
  Choosing this option sacrifices read-your-writes consistency to improve
  availability.

  Enums:
    FailoverRadiusValueValuesEnum: The failover radius of this app profile,
      determining which clusters request failovers could be routed to.
      FAILOVER_RADIUS_UNSPECIFIED is interpreted as ANY_REGION. If specified
      in addition to cluster_ids, both restrictions will be applied. For
      example, let there be four clusters in the following zones: - us-east1-b
      - us-east1-c - us-east1-d - europe-west1-b If a multi-cluster app
      profile specifies the set of cluster IDs in us-east1-b, us-east1-c, and
      europe-west1-b, requests will never arrive at or fail over to us-
      east1-d. If the app profile also specifies a INITIAL_REGION_ONLY
      failover radius, requests will, in addition, only be able to fail over
      within the region of the first cluster routed to. As an example,
      requests that are first routed to us-east1-b will only be able to fail
      over to us-east1-c (since us-east1-d is not in the set of cluster IDs
      specified and europe-west1-b is in a different region). Requests that
      are first routed to europe-west1-b will not fail over at all.

  Fields:
    clusterIds: The set of clusters to route to. The order is ignored;
      clusters will be tried in order of distance. If left empty, all clusters
      are eligible.
    failoverRadius: The failover radius of this app profile, determining which
      clusters request failovers could be routed to.
      FAILOVER_RADIUS_UNSPECIFIED is interpreted as ANY_REGION. If specified
      in addition to cluster_ids, both restrictions will be applied. For
      example, let there be four clusters in the following zones: - us-east1-b
      - us-east1-c - us-east1-d - europe-west1-b If a multi-cluster app
      profile specifies the set of cluster IDs in us-east1-b, us-east1-c, and
      europe-west1-b, requests will never arrive at or fail over to us-
      east1-d. If the app profile also specifies a INITIAL_REGION_ONLY
      failover radius, requests will, in addition, only be able to fail over
      within the region of the first cluster routed to. As an example,
      requests that are first routed to us-east1-b will only be able to fail
      over to us-east1-c (since us-east1-d is not in the set of cluster IDs
      specified and europe-west1-b is in a different region). Requests that
      are first routed to europe-west1-b will not fail over at all.
    rowAffinity: Row affinity sticky routing based on the row key of the
      request. Requests that span multiple rows are routed non-
      deterministically.
  """

    class FailoverRadiusValueValuesEnum(_messages.Enum):
        """The failover radius of this app profile, determining which clusters
    request failovers could be routed to. FAILOVER_RADIUS_UNSPECIFIED is
    interpreted as ANY_REGION. If specified in addition to cluster_ids, both
    restrictions will be applied. For example, let there be four clusters in
    the following zones: - us-east1-b - us-east1-c - us-east1-d - europe-
    west1-b If a multi-cluster app profile specifies the set of cluster IDs in
    us-east1-b, us-east1-c, and europe-west1-b, requests will never arrive at
    or fail over to us-east1-d. If the app profile also specifies a
    INITIAL_REGION_ONLY failover radius, requests will, in addition, only be
    able to fail over within the region of the first cluster routed to. As an
    example, requests that are first routed to us-east1-b will only be able to
    fail over to us-east1-c (since us-east1-d is not in the set of cluster IDs
    specified and europe-west1-b is in a different region). Requests that are
    first routed to europe-west1-b will not fail over at all.

    Values:
      FAILOVER_RADIUS_UNSPECIFIED: No failover radius specified.
      ANY_REGION: Fail over to all clusters in the instance.
      INITIAL_REGION_ONLY: Fail over only to clusters in the same region as
        the first cluster routed to.
    """
        FAILOVER_RADIUS_UNSPECIFIED = 0
        ANY_REGION = 1
        INITIAL_REGION_ONLY = 2
    clusterIds = _messages.StringField(1, repeated=True)
    failoverRadius = _messages.EnumField('FailoverRadiusValueValuesEnum', 2)
    rowAffinity = _messages.MessageField('RowAffinity', 3)