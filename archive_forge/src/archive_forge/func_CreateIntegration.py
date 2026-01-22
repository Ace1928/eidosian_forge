from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals
import contextlib
import datetime
import json
from typing import List, MutableSequence, Optional
from apitools.base.py import exceptions as api_exceptions
from googlecloudsdk.api_lib.run import global_methods
from googlecloudsdk.api_lib.run.integrations import api_utils
from googlecloudsdk.api_lib.run.integrations import types_utils
from googlecloudsdk.api_lib.run.integrations import validator
from googlecloudsdk.api_lib.util import apis
from googlecloudsdk.command_lib.run import connection_context
from googlecloudsdk.command_lib.run import flags as run_flags
from googlecloudsdk.command_lib.run import serverless_operations
from googlecloudsdk.command_lib.run.integrations import flags
from googlecloudsdk.command_lib.run.integrations import integration_list_printer
from googlecloudsdk.command_lib.run.integrations import messages_util
from googlecloudsdk.command_lib.run.integrations import stages
from googlecloudsdk.command_lib.run.integrations import typekits_util
from googlecloudsdk.command_lib.run.integrations.typekits import base
from googlecloudsdk.command_lib.runapps import exceptions
from googlecloudsdk.core import properties
from googlecloudsdk.core import resources
from googlecloudsdk.core import yaml
from googlecloudsdk.core.console import progress_tracker
from googlecloudsdk.generated_clients.apis.runapps.v1alpha1 import runapps_v1alpha1_messages
import six
def CreateIntegration(self, integration_type, parameters, service, name=None):
    """Create an integration.

    Args:
      integration_type:  type of the integration.
      parameters: parameter dictionary from args.
      service: the service to attach to the new integration.
      name: name of the integration, if empty, a defalt one will be generated.

    Returns:
      The name of the integration.
    """
    app = self.GetDefaultApp()
    typekit = typekits_util.GetTypeKit(integration_type)
    if name and typekit.is_singleton:
        raise exceptions.ArgumentError('--name is not allowed for integration type [{}].'.format(integration_type))
    if not name:
        name = typekit.NewIntegrationName(app.config)
    resource_type = typekit.resource_type
    if self._FindResource(app.config, name, resource_type):
        raise exceptions.ArgumentError(messages_util.IntegrationAlreadyExists(name))
    resource = runapps_v1alpha1_messages.Resource(id=runapps_v1alpha1_messages.ResourceID(name=name, type=resource_type))
    services_in_params = typekit.UpdateResourceConfig(parameters, resource)
    app.config.resourceList.append(resource)
    match_type_names = typekit.GetCreateSelectors(name)
    services = [service] if service else []
    services.extend(services_in_params)
    for svc in services:
        match_type_names.append({'type': types_utils.SERVICE_TYPE, 'name': svc, 'ignoreResourceConfig': True})
    for svc in services:
        self.EnsureWorkloadResources(app.config, svc, types_utils.SERVICE_TYPE)
    self.CheckCloudRunServicesExistence(services)
    if service:
        workload = self._FindResource(app.config, service, types_utils.SERVICE_TYPE)
        typekit.BindServiceToIntegration(integration=resource, workload=workload)
    resource_stages = typekit.GetCreateComponentTypes(selectors=match_type_names)
    deploy_message = typekit.GetDeployMessage(create=True)
    stages_map = stages.IntegrationStages(create=True, resource_types=resource_stages)

    def StatusUpdate(tracker, operation, unused_status):
        self._UpdateDeploymentTracker(tracker, operation, stages_map)
        return
    with progress_tracker.StagedProgressTracker('Creating new Integration...', stages_map.values(), failure_message='Failed to create new integration.') as tracker:
        try:
            self.ApplyAppConfig(tracker=tracker, tracker_update_func=StatusUpdate, appname=_DEFAULT_APP_NAME, appconfig=app.config, integration_name=name, deploy_message=deploy_message, match_type_names=match_type_names, etag=app.etag)
        except exceptions.IntegrationsOperationError as err:
            tracker.AddWarning(messages_util.RetryDeploymentMessage(self._release_track, name))
            raise err
    return name