from oslo_log import log as logging
from saharaclient.osc.v1 import images as images_v1
class ListImages(images_v1.ListImages):
    """Lists registered images"""
    log = logging.getLogger(__name__ + '.ListImages')