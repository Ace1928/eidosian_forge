from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.calliope import arg_parsers
from googlecloudsdk.calliope.base import ReleaseTrack
from googlecloudsdk.command_lib.assured import resource_args
from googlecloudsdk.command_lib.util.concepts import concept_parsers
def AddAcknowledgeViolationsFlags(parser):
    """Method to add acknowledge violations flags."""
    AddViolationResourceArgToParser(parser, verb='acknowledge')
    parser.add_argument('--comment', required=True, help='Business justification used added to acknowledge a violation.')
    parser.add_argument('--acknowledge-type', help='the acknowledge type for specified violation, which is one of:\n      SINGLE_VIOLATION - to acknowledge specified violation,\n      EXISTING_CHILD_RESOURCE_VIOLATIONS - to acknowledge specified org policy\n      violation and all associated child resource violations.')