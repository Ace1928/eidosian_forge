from oslo_log import log as logging
from saharaclient.osc.v1 import plugins as p_v1
class GetPluginConfigs(p_v1.GetPluginConfigs):
    """Get plugin configs"""
    log = logging.getLogger(__name__ + '.GetPluginConfigs')