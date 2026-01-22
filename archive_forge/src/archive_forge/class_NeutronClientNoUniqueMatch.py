from oslo_utils import encodeutils
from neutronclient._i18n import _
class NeutronClientNoUniqueMatch(NeutronCLIError):
    message = _("Multiple %(resource)s matches found for name '%(name)s', use an ID to be more specific.")