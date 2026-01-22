from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
def SetForceFieldInDeleteEntitlementRequest(unused_ref, unused_args, req):
    """Modify request hook to set the force field in delete entitlement requests to true."""
    req.force = True
    return req