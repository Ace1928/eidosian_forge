from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from apitools.base.py import exceptions as apitools_exceptions
from googlecloudsdk.api_lib.util import exceptions as api_exceptions
from googlecloudsdk.calliope import base
from googlecloudsdk.calliope import exceptions
from googlecloudsdk.core import log
from googlecloudsdk.core import yaml
from googlecloudsdk.core.resource import resource_printer
from six.moves import range  # pylint: disable=redefined-builtin
def FetchDeployment(client, messages, project, deployment_name):
    """Returns the deployment."""
    try:
        deployment_response = client.deployments.Get(messages.DeploymentmanagerDeploymentsGetRequest(project=project, deployment=deployment_name))
        return deployment_response
    except apitools_exceptions.HttpError as error:
        raise api_exceptions.HttpException(error, HTTP_ERROR_FORMAT)