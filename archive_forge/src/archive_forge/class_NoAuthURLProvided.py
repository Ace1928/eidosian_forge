from oslo_utils import encodeutils
from neutronclient._i18n import _
class NoAuthURLProvided(Unauthorized):
    message = _('auth_url was not provided to the Neutron client')