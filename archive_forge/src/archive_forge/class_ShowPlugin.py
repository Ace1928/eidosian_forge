from oslo_log import log as logging
from saharaclient.osc.v1 import plugins as p_v1
class ShowPlugin(p_v1.ShowPlugin):
    """Display plugin details"""
    log = logging.getLogger(__name__ + '.ShowPlugin')