from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.api_lib.sql import api_util
from googlecloudsdk.calliope import base
from googlecloudsdk.command_lib.sql import flags
def _AddCommonFlags(parser):
    """Adds flags common to all release tracks."""
    parser.display_info.AddFormat('\n    table(\n        name,\n        type,\n        appliesTo.list():label=DATABASE_VERSION,\n        allowedStringValues.list():label=ALLOWED_VALUES\n      )\n    ')