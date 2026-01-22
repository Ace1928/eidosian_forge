from blazarclient import client as blazar_client
from blazarclient import exception as client_exception
from oslo_config import cfg
from heat.common import exception
from heat.engine.clients import client_plugin
from heat.engine import constraints
class BlazarBaseConstraint(constraints.BaseCustomConstraint):
    resource_client_name = CLIENT_NAME