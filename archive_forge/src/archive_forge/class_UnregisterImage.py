from oslo_log import log as logging
from saharaclient.osc.v1 import images as images_v1
class UnregisterImage(images_v1.UnregisterImage):
    """Unregister image(s)"""
    log = logging.getLogger(__name__ + '.RegisterImage')