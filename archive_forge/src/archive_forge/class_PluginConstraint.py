from oslo_config import cfg
from saharaclient.api import base as sahara_base
from saharaclient import client as sahara_client
from heat.common import exception
from heat.common.i18n import _
from heat.engine.clients import client_plugin
from heat.engine import constraints
class PluginConstraint(constraints.BaseCustomConstraint):
    resource_client_name = CLIENT_NAME
    resource_getter_name = 'get_plugin_id'