from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.command_lib.compute import scope as compute_scope
def IsRegionalTargetHttpsProxiesRef(target_https_proxy_ref):
    """Returns True if the Target HTTPS Proxy reference is regional."""
    return target_https_proxy_ref.Collection() == 'compute.regionTargetHttpsProxies'