from cinderclient import client as cc
from cinderclient import exceptions
from keystoneauth1 import exceptions as ks_exceptions
from oslo_config import cfg
from oslo_log import log as logging
from heat.common import exception
from heat.common.i18n import _
from heat.engine.clients import client_plugin
from heat.engine.clients import os as os_client
from heat.engine import constraints
class QoSSpecsConstraint(BaseCinderConstraint):
    expected_exceptions = (exceptions.NotFound,)
    resource_getter_name = 'get_qos_specs'