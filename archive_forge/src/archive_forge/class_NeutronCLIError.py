from oslo_utils import encodeutils
from neutronclient._i18n import _
class NeutronCLIError(NeutronException):
    """Exception raised when command line parsing fails."""
    pass