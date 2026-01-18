import netaddr
import re
from heat.common.i18n import _
def validate_dns_format(data):
    if not data:
        return
    trimmed = data if not data.endswith('.') else data[:-1]
    if len(trimmed) > FQDN_MAX_LEN:
        raise ValueError(_("'%(data)s' exceeds the %(max_len)s character FQDN limit") % {'data': trimmed, 'max_len': FQDN_MAX_LEN})
    names = trimmed.split('.')
    for name in names:
        if not name:
            raise ValueError(_('Encountered an empty component.'))
        if name.endswith('-') or name.startswith('-'):
            raise ValueError(_("Name '%s' must not start or end with a hyphen.") % name)
        if not re.match(DNS_LABEL_REGEX, name):
            raise ValueError(_("Name '%(name)s' must be 1-%(max_len)s characters long, each of which can only be alphanumeric or a hyphen.") % {'name': name, 'max_len': DNS_LABEL_MAX_LEN})
    if data.endswith('.') and len(names) > 1 and re.match('^[0-9]+$', names[-1]):
        raise ValueError(_("TLD '%s' must not be all numeric.") % names[-1])