from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.calliope import exceptions as calliope_exceptions
from googlecloudsdk.command_lib.compute import flags as compute_flags
from googlecloudsdk.command_lib.compute import scope as compute_scope
from googlecloudsdk.command_lib.compute.networks.subnets import flags as subnet_flags
from googlecloudsdk.command_lib.compute.routers.nats import flags as nat_flags
from googlecloudsdk.core import exceptions as core_exceptions
from googlecloudsdk.core import yaml
from googlecloudsdk.core.util import files
import six
def _TranslateLogFilter(filter_str, compute_holder):
    """Translates the specified log filter to the enum value."""
    if filter_str == 'ALL':
        return compute_holder.client.messages.RouterNatLogConfig.FilterValueValuesEnum.ALL
    if filter_str == 'TRANSLATIONS_ONLY':
        return compute_holder.client.messages.RouterNatLogConfig.FilterValueValuesEnum.TRANSLATIONS_ONLY
    if filter_str == 'ERRORS_ONLY':
        return compute_holder.client.messages.RouterNatLogConfig.FilterValueValuesEnum.ERRORS_ONLY
    raise calliope_exceptions.InvalidArgumentException('--log-filter', '--log-filter must be ALL, TRANSLATIONS_ONLY or ERRORS_ONLY')