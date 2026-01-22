from ironicclient.common.apiclient import exceptions as ic_exc
from ironicclient.v1 import client as ironic_client
from oslo_config import cfg
from heat.common import exception
from heat.engine.clients import client_plugin
from heat.engine import constraints
class NodeConstraint(constraints.BaseCustomConstraint):
    resource_client_name = CLIENT_NAME
    resource_getter_name = 'get_node'