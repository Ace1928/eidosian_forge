from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.calliope import arg_parsers
import six
class InvalidTimeOfDayError(Error):
    """Error for passing invalid time of day."""