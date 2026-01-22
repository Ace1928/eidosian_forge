from openstack.clustering.v1 import _async_resource
from openstack import resource
from openstack import utils
class NodeDetail(Node):
    base_path = '/nodes/%(node_id)s?show_details=True'
    allow_create = False
    allow_fetch = True
    allow_commit = False
    allow_delete = False
    allow_list = False
    node_id = resource.URI('node_id')