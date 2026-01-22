from openstack.common import tag
from openstack import exceptions
from openstack.network.v2 import _base
from openstack import resource
from openstack import utils
class L3AgentRouter(Router):
    resource_key = 'router'
    resources_key = 'routers'
    base_path = '/agents/%(agent_id)s/l3-routers'
    resource_name = 'l3-router'
    allow_create = False
    allow_retrieve = True
    allow_commit = False
    allow_delete = False
    allow_list = True