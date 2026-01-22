from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from apitools.base.py import exceptions as apitools_exceptions
from googlecloudsdk.api_lib.deployment_manager import dm_api_util
from googlecloudsdk.api_lib.deployment_manager import dm_base
from googlecloudsdk.api_lib.deployment_manager import exceptions as dm_exceptions
from googlecloudsdk.api_lib.util import apis
from googlecloudsdk.calliope import base
from googlecloudsdk.calliope import exceptions
from googlecloudsdk.command_lib.deployment_manager import alpha_flags
from googlecloudsdk.command_lib.deployment_manager import dm_util
from googlecloudsdk.command_lib.deployment_manager import dm_write
from googlecloudsdk.command_lib.deployment_manager import flags
from googlecloudsdk.command_lib.deployment_manager import importer
from googlecloudsdk.command_lib.util.apis import arg_utils
from googlecloudsdk.command_lib.util.args import labels_util
from googlecloudsdk.core import log
from googlecloudsdk.core import properties
import six
def _PerformRollback(self, deployment_name, error_message):
    log.warning('There was an error deploying ' + deployment_name + ':\n' + error_message)
    log.status.Print('`--automatic-rollback-on-error` flag was supplied; deleting failed deployment...')
    try:
        delete_operation = self.client.deployments.Delete(self.messages.DeploymentmanagerDeploymentsDeleteRequest(project=dm_base.GetProject(), deployment=deployment_name))
    except apitools_exceptions.HttpError as error:
        raise exceptions.HttpException(error, dm_api_util.HTTP_ERROR_FORMAT)
    dm_write.WaitForOperation(self.client, self.messages, delete_operation.name, 'delete', dm_base.GetProject(), timeout=OPERATION_TIMEOUT)
    completed_operation = dm_api_util.GetOperation(self.client, self.messages, delete_operation, dm_base.GetProject())
    return completed_operation