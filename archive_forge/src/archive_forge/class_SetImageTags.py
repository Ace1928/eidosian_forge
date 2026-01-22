from oslo_log import log as logging
from saharaclient.osc.v1 import images as images_v1
class SetImageTags(images_v1.SetImageTags):
    """Set image tags (Replace current image tags with provided ones)"""
    log = logging.getLogger(__name__ + '.AddImageTags')