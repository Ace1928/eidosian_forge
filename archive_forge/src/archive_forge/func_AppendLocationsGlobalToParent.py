from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import re
from googlecloudsdk.core import exceptions
def AppendLocationsGlobalToParent(unused_ref, unused_args, request):
    """Add locations/global to parent path, since it isn't automatically populated by apitools."""
    request.parent += '/locations/global'
    return request