from oslo_utils import excutils
from neutron_lib._i18n import _
class PreexistingDeviceFailure(NeutronException):
    """A creation error due to an already existing device.

    An exception indication creation failed due to an already existing
    device.

    :param dev_name: The device name that already exists.
    """
    message = _('Creation failed. %(dev_name)s already exists.')