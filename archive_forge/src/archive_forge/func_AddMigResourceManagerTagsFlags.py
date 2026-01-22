from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import collections
import re
from typing import Any
from googlecloudsdk.calliope import arg_parsers
from googlecloudsdk.calliope import base
from googlecloudsdk.calliope import exceptions
from googlecloudsdk.command_lib.compute import flags as compute_flags
from googlecloudsdk.command_lib.util.apis import arg_utils
def AddMigResourceManagerTagsFlags(parser):
    """Adds resource manager tag related flags."""
    parser.add_argument('--resource-manager-tags', type=arg_parsers.ArgDict(), metavar='KEY=VALUE', action=arg_parsers.UpdateAction, help='Specifies a list of resource manager tags to apply to the managed instance group. A resource manager tag is a key-value pair. You can attach exactly one value to a MIG for a given key. A MIG can have a maximum of 50 key-value pairs attached.')