from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import sys
from googlecloudsdk.api_lib.compute import utils as compute_utils
from googlecloudsdk.api_lib.storage import storage_util
from googlecloudsdk.api_lib.util import apis
from googlecloudsdk.calliope import actions
from googlecloudsdk.calliope import arg_parsers
from googlecloudsdk.calliope import base
from googlecloudsdk.command_lib.util import completers
def AddLocationGroup(parser, hidden=False, specify_default_region=True):
    location_group = parser.add_mutually_exclusive_group(hidden=hidden)
    AddRegion(location_group, hidden=hidden, specify_default_region=specify_default_region)
    AddZone(location_group, help_text='Preferred Compute Engine zone (e.g. us-central1-a, us-central1-b, etc.).', hidden=hidden)