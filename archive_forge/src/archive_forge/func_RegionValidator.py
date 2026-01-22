from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.calliope import arg_parsers
from googlecloudsdk.calliope import exceptions
from googlecloudsdk.core import log
import six.moves.urllib.parse
def RegionValidator(region):
    """RegionValidator is a no-op. The validation is handled in CLH server."""
    return region