from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from apitools.base.py import list_pager
from googlecloudsdk.api_lib.apphub import consts as api_lib_consts
from googlecloudsdk.api_lib.apphub import utils as api_lib_utils
from googlecloudsdk.api_lib.util import waiter
from googlecloudsdk.calliope import base
from googlecloudsdk.core import log
def _UpdateHelper(self, args, attributes):
    """Helper to generate workload and update_mask fields for update_request."""
    workload = self.messages.Workload()
    update_mask = ''
    if args.display_name:
        update_mask = api_lib_utils.AddToUpdateMask(update_mask, api_lib_consts.UpdateApplicationWorkload.UPDATE_MASK_DISPLAY_NAME_FIELD_NAME)
        workload.displayName = args.display_name
    if args.description:
        update_mask = api_lib_utils.AddToUpdateMask(update_mask, api_lib_consts.UpdateApplicationWorkload.UPDATE_MASK_DESCRIPTION_FIELD_NAME)
        workload.description = args.description
    if attributes.criticality:
        update_mask = api_lib_utils.AddToUpdateMask(update_mask, api_lib_consts.UpdateApplicationWorkload.UPDATE_MASK_ATTR_CRITICALITY_FIELD_NAME)
    if attributes.environment:
        update_mask = api_lib_utils.AddToUpdateMask(update_mask, api_lib_consts.UpdateApplicationWorkload.UPDATE_MASK_ATTR_ENVIRONMENT_FIELD_NAME)
    if attributes.businessOwners:
        update_mask = api_lib_utils.AddToUpdateMask(update_mask, api_lib_consts.UpdateApplicationWorkload.UPDATE_MASK_ATTR_BUSINESS_OWNERS_FIELD_NAME)
    if attributes.developerOwners:
        update_mask = api_lib_utils.AddToUpdateMask(update_mask, api_lib_consts.UpdateApplicationWorkload.UPDATE_MASK_ATTR_DEVELOPER_OWNERS_FIELD_NAME)
    if attributes.operatorOwners:
        update_mask = api_lib_utils.AddToUpdateMask(update_mask, api_lib_consts.UpdateApplicationWorkload.UPDATE_MASK_ATTR_OPERATOR_OWNERS_FIELD_NAME)
    workload.attributes = attributes
    return (workload, update_mask)