from glance.api import policy
from glance.api import property_protections
from glance.common import exception
from glance.common import property_utils
import glance.domain
from glance.tests import utils
class ImageStub(object):

    def __init__(self, extra_prop):
        self.extra_properties = extra_prop