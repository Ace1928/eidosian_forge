from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.calliope import arg_parsers
def GetPlatformVersion(args):
    return getattr(args, 'platform_version', None)