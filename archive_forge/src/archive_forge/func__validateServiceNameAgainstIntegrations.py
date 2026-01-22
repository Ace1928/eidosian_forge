from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.api_lib.run.integrations import types_utils
from googlecloudsdk.calliope import base
from googlecloudsdk.command_lib.run import pretty_print
from googlecloudsdk.command_lib.run.integrations import flags
from googlecloudsdk.command_lib.run.integrations import messages_util
from googlecloudsdk.command_lib.run.integrations import run_apps_operations
from googlecloudsdk.command_lib.runapps import exceptions
def _validateServiceNameAgainstIntegrations(self, client, integration_type, integration_name, service):
    """Checks if the service name matches an integration name."""
    if not service:
        return
    error = exceptions.ArgumentError('Service name cannot be the same as ' + 'the provided integration name or an ' + 'existing integration name')
    if service == integration_name:
        raise error
    integrations = client.ListIntegrations(integration_type, None)
    for integration in integrations:
        if integration.integration_name == service:
            raise error