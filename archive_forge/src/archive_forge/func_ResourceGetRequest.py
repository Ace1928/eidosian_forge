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
def ResourceGetRequest(self):
    """"Generates apitools request message to get the resource."""
    target_link = self.operation.targetLink
    if self.project:
        request = self.resource_service.GetRequestType('Get')(project=self.project)
    else:
        if target_link is None:
            log.status.write('{0}.\n'.format(_HumanFriendlyNameForOpPastTense(self.operation.operationType).capitalize()))
            return
        token_list = target_link.split('/')
        flexible_resource_id = token_list[-1]
        request = self.resource_service.GetRequestType('Get')(securityPolicy=flexible_resource_id)
    if self.operation.zone:
        request.zone = path_simplifier.Name(self.operation.zone)
    elif self.operation.region:
        request.region = path_simplifier.Name(self.operation.region)
    resource_params = self.resource_service.GetMethodConfig('Get').ordered_params
    name_field = resource_params[-1]
    if len(resource_params) == 4:
        if self.resize_request_name:
            target_link = target_link + '/resizeRequests/' + self.resize_request_name
        parent_resource_field = resource_params[2]
        parent_resource_name = target_link.split('/')[-3]
        setattr(request, parent_resource_field, parent_resource_name)
    resource_name = self.followup_override or path_simplifier.Name(target_link)
    setattr(request, name_field, resource_name)
    return request