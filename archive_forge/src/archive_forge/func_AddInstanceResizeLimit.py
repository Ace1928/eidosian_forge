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
def AddInstanceResizeLimit(parser):
    parser.add_argument('--storage-auto-increase-limit', type=arg_parsers.BoundedInt(10, sys.maxsize, unlimited=True), help='Allows you to set a maximum storage capacity, in GB. Automatic increases to your capacity will stop once this limit has been reached. Default capacity is *unlimited*.')