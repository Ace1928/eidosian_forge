from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import textwrap
from googlecloudsdk.calliope import arg_parsers
from googlecloudsdk.calliope import base
def AddDisableInitialRolloutFlag():
    """Adds --disable-initial-rollout flag."""
    return base.Argument('--disable-initial-rollout', action='store_const', help='Skips creating a rollout in the first target defined in the delivery pipeline.', const=True)