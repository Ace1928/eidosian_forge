from oslo_config import cfg
from oslo_log import log as logging
from heat.common.i18n import _
from heat.engine import attributes
from heat.engine import constraints
from heat.engine import properties
from heat.engine.resources import server_base
from heat.engine import support
def user_data_software_config(self):
    return True