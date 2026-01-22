from openstack import exceptions
from openstack import resource
from openstack import utils
class RouterL3Agent(Agent):
    resource_key = 'agent'
    resources_key = 'agents'
    base_path = '/routers/%(router_id)s/l3-agents'
    resource_name = 'l3-agent'
    allow_create = False
    allow_retrieve = True
    allow_commit = False
    allow_delete = False
    allow_list = True