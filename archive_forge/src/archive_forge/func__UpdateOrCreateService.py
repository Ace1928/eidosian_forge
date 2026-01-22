from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals
import contextlib
import dataclasses
import functools
import random
import string
from apitools.base.py import encoding
from apitools.base.py import exceptions as api_exceptions
from apitools.base.py import list_pager
from googlecloudsdk.api_lib.run import condition as run_condition
from googlecloudsdk.api_lib.run import configuration
from googlecloudsdk.api_lib.run import domain_mapping
from googlecloudsdk.api_lib.run import execution
from googlecloudsdk.api_lib.run import global_methods
from googlecloudsdk.api_lib.run import job
from googlecloudsdk.api_lib.run import metric_names
from googlecloudsdk.api_lib.run import revision
from googlecloudsdk.api_lib.run import route
from googlecloudsdk.api_lib.run import service
from googlecloudsdk.api_lib.run import task
from googlecloudsdk.api_lib.util import apis
from googlecloudsdk.api_lib.util import apis_internal
from googlecloudsdk.api_lib.util import waiter
from googlecloudsdk.command_lib.iam import iam_util
from googlecloudsdk.command_lib.run import artifact_registry
from googlecloudsdk.command_lib.run import config_changes as config_changes_mod
from googlecloudsdk.command_lib.run import exceptions as serverless_exceptions
from googlecloudsdk.command_lib.run import messages_util
from googlecloudsdk.command_lib.run import name_generator
from googlecloudsdk.command_lib.run import op_pollers
from googlecloudsdk.command_lib.run import resource_name_conversion
from googlecloudsdk.command_lib.run import stages
from googlecloudsdk.command_lib.run.sourcedeploys import deployer
from googlecloudsdk.core import exceptions
from googlecloudsdk.core import log
from googlecloudsdk.core import metrics
from googlecloudsdk.core import properties
from googlecloudsdk.core import resources
from googlecloudsdk.core.console import progress_tracker
from googlecloudsdk.core.util import retry
import six
def _UpdateOrCreateService(self, service_ref, config_changes, with_code, serv, dry_run=False):
    """Apply config_changes to the service.

    Create it if necessary.

    Arguments:
      service_ref: Reference to the service to create or update
      config_changes: list of ConfigChanger to modify the service with
      with_code: bool, True if the config_changes contains code to deploy. We
        can't create the service if we're not deploying code.
      serv: service.Service, For update the Service to update and for create
        None.
      dry_run: bool, if True only validate the change.

    Returns:
      The Service object we created or modified.
    """
    messages = self.messages_module
    try:
        if serv:
            serv = config_changes_mod.WithChanges(serv, config_changes)
            serv_name = service_ref.RelativeName()
            serv_update_req = messages.RunNamespacesServicesReplaceServiceRequest(service=serv.Message(), name=serv_name, dryRun='all' if dry_run else None)
            with metrics.RecordDuration(metric_names.UPDATE_SERVICE):
                updated = self._client.namespaces_services.ReplaceService(serv_update_req)
            return service.Service(updated, messages)
        else:
            if not with_code:
                raise serverless_exceptions.ServiceNotFoundError('Service [{}] could not be found.'.format(service_ref.servicesId))
            new_serv = service.Service.New(self._client, service_ref.namespacesId)
            new_serv.name = service_ref.servicesId
            parent = service_ref.Parent().RelativeName()
            new_serv = config_changes_mod.WithChanges(new_serv, config_changes)
            serv_create_req = messages.RunNamespacesServicesCreateRequest(service=new_serv.Message(), parent=parent, dryRun='all' if dry_run else None)
            with metrics.RecordDuration(metric_names.CREATE_SERVICE):
                raw_service = self._client.namespaces_services.Create(serv_create_req)
            return service.Service(raw_service, messages)
    except api_exceptions.InvalidDataFromServerError as e:
        serverless_exceptions.MaybeRaiseCustomFieldMismatch(e)
    except api_exceptions.HttpBadRequestError as e:
        exceptions.reraise(serverless_exceptions.HttpError(e))
    except api_exceptions.HttpNotFoundError as e:
        platform = properties.VALUES.run.platform.Get()
        error_msg = 'Deployment endpoint was not found.'
        if platform == 'gke':
            all_clusters = global_methods.ListClusters()
            clusters = ['* {} in {}'.format(c.name, c.zone) for c in all_clusters]
            error_msg += ' Perhaps the provided cluster was invalid or does not have Cloud Run enabled. Pass the `--cluster` and `--cluster-location` flags or set the `run/cluster` and `run/cluster_location` properties to a valid cluster and zone and retry.\nAvailable clusters:\n{}'.format('\n'.join(clusters))
        elif platform == 'managed':
            all_regions = global_methods.ListRegions(self._op_client)
            if self._region not in all_regions:
                regions = ['* {}'.format(r) for r in all_regions]
                error_msg += ' The provided region was invalid. Pass the `--region` flag or set the `run/region` property to a valid region and retry.\nAvailable regions:\n{}'.format('\n'.join(regions))
        elif platform == 'kubernetes':
            error_msg += ' Perhaps the provided cluster was invalid or does not have Cloud Run enabled. Ensure in your kubeconfig file that the cluster referenced in the current context or the specified context is a valid cluster and retry.'
        raise serverless_exceptions.DeploymentFailedError(error_msg)
    except api_exceptions.HttpError as e:
        platform = properties.VALUES.run.platform.Get()
        if platform == 'managed':
            exceptions.reraise(e)
        k8s_error = serverless_exceptions.KubernetesExceptionParser(e)
        causes = '\n\n'.join([c['message'] for c in k8s_error.causes])
        if not causes:
            causes = k8s_error.error
        raise serverless_exceptions.KubernetesError('Error{}:\n{}\n'.format('s' if len(k8s_error.causes) > 1 else '', causes))