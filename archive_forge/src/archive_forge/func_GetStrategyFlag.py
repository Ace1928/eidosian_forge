from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.calliope import arg_parsers
from googlecloudsdk.calliope import base
from googlecloudsdk.core.util import files
def GetStrategyFlag():
    return base.Argument('--strategy', required=False, choices=_MERGE_STRATEGIES, help='Controls how changes to the local package are handled.')