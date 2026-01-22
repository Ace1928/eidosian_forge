from oslo_utils import excutils
from neutron_lib._i18n import _
class ResourceExhausted(ServiceUnavailable):
    """A service unavailable error indicating a resource is exhausted."""
    pass