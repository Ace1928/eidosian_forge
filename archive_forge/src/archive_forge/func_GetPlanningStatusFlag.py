from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.calliope import arg_parsers
from googlecloudsdk.calliope import base
from googlecloudsdk.command_lib.compute.instances import flags as instance_flags
from googlecloudsdk.command_lib.compute.reservations import flags as reservation_flags
def GetPlanningStatusFlag():
    """Gets the --planning-status flag."""
    help_text = "  The planning status of the future reservation. The default value is DRAFT.\n  While in DRAFT, any changes to the future reservation's properties will be\n  allowed. If set to SUBMITTED, the future reservation will submit and its\n  procurementStatus will change to PENDING_APPROVAL. Once the future reservation\n  is pending approval, changes to the future reservation's properties will not\n  be allowed.\n  "
    return base.Argument('--planning-status', type=lambda x: x.upper(), choices={'DRAFT': 'Default planning status value.', 'SUBMITTED': 'Planning status value to immediately submit the future reservation.'}, help=help_text)