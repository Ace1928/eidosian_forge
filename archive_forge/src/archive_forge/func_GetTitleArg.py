from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import re
from googlecloudsdk.api_lib.util import waiter
from googlecloudsdk.calliope import base
from googlecloudsdk.core import exceptions
def GetTitleArg(noun):
    return base.Argument('--title', help='Short human-readable title of the {}.'.format(noun))