from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import os
import sys
import uuid
from apitools.base.py import encoding
from apitools.base.py import exceptions as apitools_exceptions
from apitools.base.py import extra_types
from googlecloudsdk.api_lib.infra_manager import configmanager_util
from googlecloudsdk.api_lib.storage import storage_api
from googlecloudsdk.api_lib.storage import storage_util
from googlecloudsdk.calliope import exceptions as c_exceptions
from googlecloudsdk.command_lib.infra_manager import deterministic_snapshot
from googlecloudsdk.command_lib.infra_manager import errors
from googlecloudsdk.command_lib.infra_manager import staging_bucket_util
from googlecloudsdk.core import log
from googlecloudsdk.core import requests
from googlecloudsdk.core import resources
from googlecloudsdk.core.resource import resource_transform
from googlecloudsdk.core.util import files
from googlecloudsdk.core.util import times
import six
def LockDeployment(messages, async_, deployment_full_name):
    """Locks the deployment.

  Args:
    messages: ModuleType, the messages module that lets us form Infra Manager
      API messages based on our protos.
    async_: bool, if True, gcloud will return immediately, otherwise it will
      wait on the long-running operation.
    deployment_full_name: string, the fully qualified name of the deployment,
      e.g. "projects/p/locations/l/deployments/d".

  Returns:
    A lock info resource or, in case async_ is True, a
      long-running operation.
  """
    lock_deployment_request = messages.LockDeploymentRequest()
    op = configmanager_util.LockDeployment(lock_deployment_request, deployment_full_name)
    deployment_ref = resources.REGISTRY.Parse(deployment_full_name, collection='config.projects.locations.deployments')
    deployment_id = deployment_ref.Name()
    log.debug('LRO: %s', op.name)
    if async_:
        log.status.Print('Lock deployment request issued for: [{0}]'.format(deployment_id))
        log.status.Print('Check operation [{}] for status.'.format(op.name))
        return op
    progress_message = 'Locking the deployment'
    lock_response = configmanager_util.WaitForApplyDeploymentOperation(op, progress_message)
    if lock_response.lockState == messages.Deployment.LockStateValueValuesEnum.LOCK_FAILED:
        raise errors.OperationFailedError('Lock deployment operation failed.')
    return ExportLock(deployment_full_name)