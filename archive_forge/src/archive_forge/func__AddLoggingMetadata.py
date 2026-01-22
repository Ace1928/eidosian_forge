from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from apitools.base.py import encoding
from googlecloudsdk.api_lib.compute import base_classes
from googlecloudsdk.api_lib.compute import firewalls_utils
from googlecloudsdk.api_lib.compute import utils as compute_api
from googlecloudsdk.api_lib.util import apis
from googlecloudsdk.calliope import base
from googlecloudsdk.calliope import exceptions as calliope_exceptions
from googlecloudsdk.command_lib.compute import exceptions
from googlecloudsdk.command_lib.compute.firewall_rules import flags
def _AddLoggingMetadata(self, messages, args, log_config):
    if args.IsSpecified('logging_metadata'):
        if log_config is None or not log_config.enable:
            raise calliope_exceptions.InvalidArgumentException('--logging-metadata', 'cannot toggle logging metadata if logging is not enabled.')
        log_config.metadata = flags.GetLoggingMetadataArg(messages).GetEnumForChoice(args.logging_metadata)