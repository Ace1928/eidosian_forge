from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import re
from apitools.base.py import exceptions as apitools_exceptions
from googlecloudsdk.api_lib.compute import base_classes
from googlecloudsdk.api_lib.iam import util as iam_api
from googlecloudsdk.api_lib.resource_manager import tags
from googlecloudsdk.api_lib.resource_manager.exceptions import ResourceManagerError
from googlecloudsdk.calliope import exceptions
from googlecloudsdk.command_lib.iam import iam_util
from googlecloudsdk.command_lib.resource_manager import endpoint_utils as endpoints
from googlecloudsdk.core import exceptions as core_exceptions
def GetNamespacedResource(namespaced_name, resource_type):
    """Gets the resource from the namespaced name.

  Args:
    namespaced_name: The namespaced name of the resource such as
      {parent_id}/{tag_key_short_name} or
      {parent_id}/{tag_key_short_name}/{tag_value_short_name}
    resource_type: the type of the resource i.e: tag_utils.TAG_KEYS,
      tag_utils.TAG_VALUES. Used to determine which service to use and which GET
      request to construct

  Returns:
    resource
  """
    with endpoints.CrmEndpointOverrides('global'):
        service = Services[resource_type]()
        req = GetByNamespacedNameRequests[resource_type](name=namespaced_name)
        response = service.GetNamespaced(req)
        return response