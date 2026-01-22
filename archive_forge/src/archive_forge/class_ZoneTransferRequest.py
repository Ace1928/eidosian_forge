from openstack.dns.v2 import _base
from openstack import resource
class ZoneTransferRequest(ZoneTransferBase):
    """DNS Zone Transfer Request Resource"""
    base_path = '/zones/tasks/transfer_requests'
    resources_key = 'transfer_requests'
    allow_create = True
    allow_fetch = True
    allow_delete = True
    allow_list = True
    allow_commit = True
    description = resource.Body('description')
    target_project_id = resource.Body('target_project_id')
    zone_name = resource.Body('zone_name')