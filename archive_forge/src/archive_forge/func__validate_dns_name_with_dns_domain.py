import re
from oslo_config import cfg
from neutron_lib._i18n import _
from neutron_lib.api import validators
from neutron_lib import constants
from neutron_lib.db import constants as db_constants
def _validate_dns_name_with_dns_domain(request_dns_name, dns_domain):
    higher_labels = dns_domain
    if dns_domain:
        higher_labels = '.%s' % dns_domain
    higher_labels_len = len(higher_labels)
    dns_name_len = len(request_dns_name)
    if not request_dns_name.endswith('.'):
        if dns_name_len + higher_labels_len > db_constants.FQDN_FIELD_SIZE:
            msg = _("The dns_name passed is a PQDN and its size is '%(dns_name_len)s'. The dns_domain option in neutron.conf is set to %(dns_domain)s, with a length of '%(higher_labels_len)s'. When the two are concatenated to form a FQDN (with a '.' at the end), the resulting length exceeds the maximum size of '%(fqdn_max_len)s'") % {'dns_name_len': dns_name_len, 'dns_domain': cfg.CONF.dns_domain, 'higher_labels_len': higher_labels_len, 'fqdn_max_len': db_constants.FQDN_FIELD_SIZE}
            return msg
        return
    if dns_name_len <= higher_labels_len or not request_dns_name.endswith(higher_labels):
        msg = _("The dns_name passed is a FQDN. Its higher level labels must be equal to the dns_domain option in neutron.conf, that has been set to '%(dns_domain)s'. It must also include one or more valid DNS labels to the left of '%(dns_domain)s'") % {'dns_domain': cfg.CONF.dns_domain}
        return msg