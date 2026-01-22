from oslo_utils import excutils
from neutron_lib._i18n import _
class DriverCallError(MultipleExceptions):

    def __init__(self, exc_list=None):
        super().__init__(exc_list or [])