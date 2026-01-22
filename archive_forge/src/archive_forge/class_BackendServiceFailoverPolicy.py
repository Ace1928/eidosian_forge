from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class BackendServiceFailoverPolicy(_messages.Message):
    """For load balancers that have configurable failover: [Internal
  passthrough Network Load Balancers](https://cloud.google.com/load-
  balancing/docs/internal/failover-overview) and [external passthrough Network
  Load Balancers](https://cloud.google.com/load-
  balancing/docs/network/networklb-failover-overview). On failover or
  failback, this field indicates whether connection draining will be honored.
  Google Cloud has a fixed connection draining timeout of 10 minutes. A
  setting of true terminates existing TCP connections to the active pool
  during failover and failback, immediately draining traffic. A setting of
  false allows existing TCP connections to persist, even on VMs no longer in
  the active pool, for up to the duration of the connection draining timeout
  (10 minutes).

  Fields:
    disableConnectionDrainOnFailover: This can be set to true only if the
      protocol is TCP. The default is false.
    dropTrafficIfUnhealthy: If set to true, connections to the load balancer
      are dropped when all primary and all backup backend VMs are unhealthy.If
      set to false, connections are distributed among all primary VMs when all
      primary and all backup backend VMs are unhealthy. For load balancers
      that have configurable failover: [Internal passthrough Network Load
      Balancers](https://cloud.google.com/load-
      balancing/docs/internal/failover-overview) and [external passthrough
      Network Load Balancers](https://cloud.google.com/load-
      balancing/docs/network/networklb-failover-overview). The default is
      false.
    failoverRatio: The value of the field must be in the range [0, 1]. If the
      value is 0, the load balancer performs a failover when the number of
      healthy primary VMs equals zero. For all other values, the load balancer
      performs a failover when the total number of healthy primary VMs is less
      than this ratio. For load balancers that have configurable failover:
      [Internal TCP/UDP Load Balancing](https://cloud.google.com/load-
      balancing/docs/internal/failover-overview) and [external TCP/UDP Load
      Balancing](https://cloud.google.com/load-
      balancing/docs/network/networklb-failover-overview).
  """
    disableConnectionDrainOnFailover = _messages.BooleanField(1)
    dropTrafficIfUnhealthy = _messages.BooleanField(2)
    failoverRatio = _messages.FloatField(3, variant=_messages.Variant.FLOAT)