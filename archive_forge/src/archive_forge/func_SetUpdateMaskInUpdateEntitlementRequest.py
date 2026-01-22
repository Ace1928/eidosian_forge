from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
def SetUpdateMaskInUpdateEntitlementRequest(unused_ref, unused_args, req):
    """Modify request hook to set the update mask field in update entitlement requests to '*'."""
    req.updateMask = '*'
    return req