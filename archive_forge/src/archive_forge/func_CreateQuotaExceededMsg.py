from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import argparse
import io
import re
from googlecloudsdk.api_lib.compute import constants
from googlecloudsdk.api_lib.compute import exceptions
from googlecloudsdk.calliope import exceptions as calliope_exceptions
from googlecloudsdk.command_lib.compute import exceptions as compute_exceptions
from googlecloudsdk.core import log
from googlecloudsdk.core.console import console_io
from googlecloudsdk.core.resource import resource_printer
import ipaddr
import six
def CreateQuotaExceededMsg(error):
    """Constructs message to show for quota exceeded error."""
    if not hasattr(error, 'errorDetails') or not error.errorDetails or (not error.errorDetails[0].quotaInfo):
        return error.message
    details = error.errorDetails[0].quotaInfo
    msg = '{}\n\tmetric name = {}\n\tlimit name = {}\n\tlimit = {}\n'.format(error.message, details.metricName, details.limitName, details.limit)
    if hasattr(details, 'futureLimit') and details.futureLimit:
        msg += '\tfuture limit = {}\n\trollout status = {}\n'.format(details.futureLimit, 'in progress')
    if details.dimensions:
        dim = io.StringIO()
        resource_printer.Print(details.dimensions, 'yaml', out=dim)
        msg += '\tdimensions = {}'.format(dim.getvalue())
    if hasattr(details, 'futureLimit') and details.futureLimit:
        msg += 'The future limit is the new default quota that will be available after a service rollout completes. For more about the rollout process, see the documentation: https://cloud.google.com/compute/docs/quota-rollout.'
    else:
        msg += 'Try your request in another zone, or view documentation on how to increase quotas: https://cloud.google.com/compute/quotas.'
    return msg