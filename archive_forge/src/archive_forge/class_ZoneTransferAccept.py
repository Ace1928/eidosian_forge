from openstack.dns.v2 import _base
from openstack import resource
class ZoneTransferAccept(ZoneTransferBase):
    """DNS Zone Transfer Accept Resource"""
    base_path = '/zones/tasks/transfer_accepts'
    resources_key = 'transfer_accepts'
    allow_create = True
    allow_fetch = True
    allow_list = True
    zone_transfer_request_id = resource.Body('zone_transfer_request_id')