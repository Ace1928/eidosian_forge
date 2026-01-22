from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from apitools.base.py import exceptions as apitools_exceptions
from googlecloudsdk.api_lib.compute import batch_helper
from googlecloudsdk.api_lib.compute import path_simplifier
from googlecloudsdk.api_lib.compute import single_request_helper
from googlecloudsdk.api_lib.util import exceptions as http_exceptions
from googlecloudsdk.command_lib.util import time_util
from googlecloudsdk.core import log
from googlecloudsdk.core import properties
def _OperationRequest(self, verb):
    """Generates apitools request message to poll the operation."""
    if self.project:
        request = self.operation_service.GetRequestType(verb)(operation=self.operation.name, project=self.project)
    else:
        token_list = self.operation.name.split('-')
        parent_id = 'organizations/' + token_list[1]
        request = self.operation_service.GetRequestType(verb)(operation=self.operation.name, parentId=parent_id)
    if self.operation.zone:
        request.zone = path_simplifier.Name(self.operation.zone)
    elif self.operation.region:
        request.region = path_simplifier.Name(self.operation.region)
    return request