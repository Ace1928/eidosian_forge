from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import re
from googlecloudsdk.calliope import parser_errors
from googlecloudsdk.core import properties
from googlecloudsdk.core.cache import resource_cache
from googlecloudsdk.core.resource import resource_property
def GetFlagFromDest(dest):
    """Returns a conventional flag name given a dest name."""
    return '--' + dest.replace('_', '-')