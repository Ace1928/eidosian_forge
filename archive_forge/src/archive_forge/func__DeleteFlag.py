from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals
from googlecloudsdk.calliope import base
from googlecloudsdk.command_lib.kuberun import flags
from googlecloudsdk.command_lib.kuberun import kuberun_command
def _DeleteFlag():
    return flags.BasicFlag('--delete', help='Delete the deployed Component from the active Environment. This can only be used to delete Components deployed in development mode. This does not modify or remove any configuration or references to the component.', required=False)