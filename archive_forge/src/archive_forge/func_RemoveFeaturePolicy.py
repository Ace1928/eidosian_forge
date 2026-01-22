from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
def RemoveFeaturePolicy(ref, _, request):
    """Remove feature policy."""
    del ref
    request.updateMask = 'featurePolicy'
    return request