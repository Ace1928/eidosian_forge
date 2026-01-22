from oslo_config import cfg
from keystone.conf import constants
from keystone.conf import utils
def _register_auth_plugin_opt(conf, option):
    conf.register_opt(option, group=GROUP_NAME)