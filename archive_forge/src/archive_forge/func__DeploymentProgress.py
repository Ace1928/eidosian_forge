from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals
from typing import Optional
from frozendict import frozendict
from googlecloudsdk.api_lib.run.integrations import types_utils
from googlecloudsdk.command_lib.run.integrations import deployment_states
from googlecloudsdk.command_lib.run.integrations.formatters import base
from googlecloudsdk.command_lib.run.integrations.formatters import custom_domains_formatter
from googlecloudsdk.command_lib.run.integrations.formatters import default_formatter
from googlecloudsdk.command_lib.run.integrations.formatters import states
from googlecloudsdk.core.console import console_attr
from googlecloudsdk.core.resource import custom_printer_base as cp
from googlecloudsdk.generated_clients.apis.runapps.v1alpha1 import runapps_v1alpha1_messages as runapps
def _DeploymentProgress(self, deployment: runapps.Deployment, formatter: base.BaseFormatter) -> str:
    """Returns a message denoting the deployment progress.

    If there is no ongoing deployment and the deployment was successful, then
    this will be empty.

    Currently this only shows something if the latest deployment was a failure.
    In the future this will be updated to show more granular statuses as the
    deployment is ongoing.

    Args:
      deployment:  The deployment object
      formatter: The specific formatter used for the integration type.

    Returns:
      The message denoting the most recent deployment's progress (failure).
    """
    if deployment is None:
        return ''
    state = str(deployment.status.state)
    if state == deployment_states.FAILED:
        reason = deployment.status.errorMessage
        symbol = formatter.StatusSymbolAndColor(states.FAILED)
        return '{} Latest deployment: FAILED - {}\n'.format(symbol, reason)
    return ''