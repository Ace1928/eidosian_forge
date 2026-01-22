from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.command_lib.compute import scope as compute_scope
def IsGlobalTargetHttpProxiesRef(target_http_proxy_ref):
    """Returns True if the Target HTTP Proxy reference is global."""
    return target_http_proxy_ref.Collection() == 'compute.targetHttpProxies'