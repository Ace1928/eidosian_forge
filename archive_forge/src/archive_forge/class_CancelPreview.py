from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from apitools.base.py import exceptions as apitools_exceptions
from googlecloudsdk.api_lib.deployment_manager import dm_api_util
from googlecloudsdk.api_lib.deployment_manager import dm_base
from googlecloudsdk.calliope import base
from googlecloudsdk.calliope import exceptions
from googlecloudsdk.command_lib.deployment_manager import dm_util
from googlecloudsdk.command_lib.deployment_manager import dm_write
from googlecloudsdk.command_lib.deployment_manager import flags
@dm_base.UseDmApi(dm_base.DmApiVersion.V2)
class CancelPreview(base.Command, dm_base.DmCommand):
    """Cancel a pending or running deployment preview.

  This command will cancel a currently running or pending preview operation on
  a deployment.
  """
    detailed_help = {'EXAMPLES': '\nTo cancel a running operation on a deployment, run:\n\n  $ {command} my-deployment\n\nTo issue a cancel preview command without waiting for the operation to complete, run:\n\n  $ {command} my-deployment --async\n\nTo cancel a preview command providing a fingerprint:\n\n  $ {command} my-deployment --fingerprint=deployment-fingerprint\n\nWhen a deployment preview is cancelled, the deployment itself is not\ndeleted.\n'}

    @staticmethod
    def Args(parser):
        """Args is called by calliope to gather arguments for this command.

    Args:
      parser: An argparse parser that you can use to add arguments that go
          on the command line after this command. Positional arguments are
          allowed.
    """
        flags.AddDeploymentNameFlag(parser)
        flags.AddAsyncFlag(parser)
        flags.AddFingerprintFlag(parser)

    def Run(self, args):
        """Run 'deployments cancel-preview'.

    Args:
      args: argparse.Namespace, The arguments that this command was invoked
          with.

    Returns:
      If --async=true, returns Operation to poll.
      Else, returns boolean indicating whether cancel preview operation
      succeeded.

    Raises:
      HttpException: An http error response was received while executing api
          request.
    """
        if args.fingerprint:
            fingerprint = dm_util.DecodeFingerprint(args.fingerprint)
        else:
            fingerprint = dm_api_util.FetchDeploymentFingerprint(self.client, self.messages, dm_base.GetProject(), args.deployment_name) or b''
        try:
            operation = self.client.deployments.CancelPreview(self.messages.DeploymentmanagerDeploymentsCancelPreviewRequest(project=dm_base.GetProject(), deployment=args.deployment_name, deploymentsCancelPreviewRequest=self.messages.DeploymentsCancelPreviewRequest(fingerprint=fingerprint)))
            new_fingerprint = dm_api_util.FetchDeploymentFingerprint(self.client, self.messages, dm_base.GetProject(), args.deployment_name)
            dm_util.PrintFingerprint(new_fingerprint)
        except apitools_exceptions.HttpError as error:
            raise exceptions.HttpException(error, dm_api_util.HTTP_ERROR_FORMAT)
        if args.async_:
            return operation
        else:
            op_name = operation.name
            try:
                operation = dm_write.WaitForOperation(self.client, self.messages, op_name, 'cancel-preview', dm_base.GetProject(), timeout=OPERATION_TIMEOUT)
                dm_util.LogOperationStatus(operation, 'Cancel preview')
            except apitools_exceptions.HttpError as error:
                raise exceptions.HttpException(error, dm_api_util.HTTP_ERROR_FORMAT)
            try:
                response = self.client.resources.List(self.messages.DeploymentmanagerResourcesListRequest(project=dm_base.GetProject(), deployment=args.deployment_name))
                return response.resources if response.resources else []
            except apitools_exceptions.HttpError as error:
                raise exceptions.HttpException(error, dm_api_util.HTTP_ERROR_FORMAT)