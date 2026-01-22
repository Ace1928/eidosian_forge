from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.api_lib.eventarc import triggers
from googlecloudsdk.calliope import base
from googlecloudsdk.calliope import exceptions
from googlecloudsdk.command_lib.eventarc import flags
from googlecloudsdk.command_lib.eventarc import types
from googlecloudsdk.core import log
def GetWorkflowDestination(self, args, old_trigger):
    if args.IsSpecified('destination_workflow'):
        return args.destination_workflow
    if old_trigger.destination.workflow:
        return old_trigger.destination.workflow.split('/')[5]
    raise exceptions.InvalidArgumentException('--destination-workflow-location', 'The specified trigger is not for a workflow destination.')