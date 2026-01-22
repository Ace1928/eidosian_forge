from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import json
from apitools.base.py import exceptions as apitools_exceptions
from googlecloudsdk.api_lib.firebase.test import exceptions
from googlecloudsdk.api_lib.util import apis
from googlecloudsdk.calliope import exceptions as calliope_exceptions
from googlecloudsdk.core import log
from googlecloudsdk.core import properties
def GetDeviceIpBlocks(context=None):
    """Gets the device IP block catalog from the TestEnvironmentDiscoveryService.

  Args:
    context: {str:object}, The current context, which is a set of key-value
      pairs that can be used for common initialization among commands.

  Returns:
    The device IP block catalog

  Raises:
    calliope_exceptions.HttpException: If it could not connect to the service.
  """
    if context:
        client = context['testing_client']
        messages = context['testing_messages']
    else:
        client = apis.GetClientInstance('testing', 'v1')
        messages = apis.GetMessagesModule('testing', 'v1')
    env_type = messages.TestingTestEnvironmentCatalogGetRequest.EnvironmentTypeValueValuesEnum.DEVICE_IP_BLOCKS
    return _GetCatalog(client, messages, env_type).deviceIpBlockCatalog