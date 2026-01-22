from oslo_log import log as logging
from saharaclient.osc.v1 import plugins as p_v1
class UpdatePlugin(p_v1.UpdatePlugin):
    log = logging.getLogger(__name__ + '.UpdatePlugin')