from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import re
from googlecloudsdk.api_lib.util import waiter
from googlecloudsdk.calliope import base
from googlecloudsdk.core import exceptions
def GetDescriptionArg(noun):
    return base.Argument('--description', help='Long-form description of {}.'.format(noun))