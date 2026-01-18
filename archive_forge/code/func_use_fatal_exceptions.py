from oslo_utils import excutils
from neutron_lib._i18n import _
def use_fatal_exceptions(self):
    """Is the instance using fatal exceptions.

        :returns: Always returns False.
        """
    return False