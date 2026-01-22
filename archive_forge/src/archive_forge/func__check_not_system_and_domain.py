from keystoneclient import base
from keystoneclient import exceptions
from keystoneclient.i18n import _
def _check_not_system_and_domain(self, system, domain):
    if system and domain:
        msg = _('Specify either system or domain, not both')
        raise exceptions.ValidationError(msg)