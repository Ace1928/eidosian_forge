from oslo_log import log as logging
from saharaclient.osc.v1 import images as images_v1
class AddImageTags(images_v1.AddImageTags):
    """Add image tags"""
    log = logging.getLogger(__name__ + '.AddImageTags')