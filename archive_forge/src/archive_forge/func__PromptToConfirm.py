from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.api_lib.compute import base_classes
from googlecloudsdk.calliope import base
from googlecloudsdk.calliope import exceptions as exceptions
from googlecloudsdk.command_lib.compute import exceptions as compute_exceptions
from googlecloudsdk.command_lib.compute.networks.subnets import flags
from googlecloudsdk.core.console import console_io
import ipaddress
import six
def _PromptToConfirm(self, subnetwork_name, original_ip_cidr_range, new_ip_cidr_range):
    prompt_message_template = 'The IP range of subnetwork [{0}] will be expanded from {1} to {2}. This operation may take several minutes to complete and cannot be undone.'
    prompt_message = prompt_message_template.format(subnetwork_name, original_ip_cidr_range, new_ip_cidr_range)
    if not console_io.PromptContinue(message=prompt_message, default=True):
        raise compute_exceptions.AbortedError('Operation aborted by user.')