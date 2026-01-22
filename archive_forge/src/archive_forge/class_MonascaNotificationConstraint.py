from monascaclient import exc as monasca_exc
from monascaclient.v2_0 import client as monasca_client
from heat.common import exception as heat_exc
from heat.engine.clients import client_plugin
from heat.engine import constraints
class MonascaNotificationConstraint(constraints.BaseCustomConstraint):
    resource_client_name = CLIENT_NAME
    resource_getter_name = 'get_notification'