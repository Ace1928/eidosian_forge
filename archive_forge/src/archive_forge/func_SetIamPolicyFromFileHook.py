from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import datetime
import re
from apitools.base.py.exceptions import HttpForbiddenError
from googlecloudsdk.api_lib.cloudresourcemanager import organizations
from googlecloudsdk.api_lib.cloudresourcemanager import projects_api
from googlecloudsdk.api_lib.cloudresourcemanager import projects_util
from googlecloudsdk.api_lib.iam import policies
from googlecloudsdk.api_lib.resource_manager import folders
from googlecloudsdk.command_lib.iam import iam_util
from googlecloudsdk.command_lib.projects import exceptions
from googlecloudsdk.core import resources
import six
def SetIamPolicyFromFileHook(ref, args, request):
    """Hook to perserve SetIAMPolicy behavior for declarative surface."""
    del ref
    del args
    update_mask = request.setIamPolicyRequest.updateMask
    if update_mask:
        mask_fields = update_mask.split(',')
        if 'bindings' not in mask_fields:
            mask_fields.append('bindings')
        if 'etag' not in update_mask:
            mask_fields.append('etag')
        request.setIamPolicyRequest.updateMask = ','.join(mask_fields)
    return request