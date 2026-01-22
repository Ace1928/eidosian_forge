import os
import re
import six
from six.moves import urllib
from routes import request_config
def _subdomain_check(kargs, mapper, environ):
    """Screen the kargs for a subdomain and alter it appropriately depending
    on the current subdomain or lack therof."""
    if mapper.sub_domains:
        subdomain = kargs.pop('sub_domain', None)
        if isinstance(subdomain, six.text_type):
            subdomain = str(subdomain)
        fullhost = environ.get('HTTP_HOST') or environ.get('SERVER_NAME')
        if not fullhost:
            return kargs
        hostmatch = fullhost.split(':')
        host = hostmatch[0]
        port = ''
        if len(hostmatch) > 1:
            port += ':' + hostmatch[1]
        match = re.match('^(.+?)\\.(%s)$' % mapper.domain_match, host)
        host_subdomain, domain = match.groups() if match else (None, host)
        subdomain = as_unicode(subdomain, mapper.encoding)
        if subdomain and host_subdomain != subdomain and (subdomain not in mapper.sub_domains_ignore):
            kargs['_host'] = subdomain + '.' + domain + port
        elif (subdomain in mapper.sub_domains_ignore or subdomain is None) and domain != host:
            kargs['_host'] = domain + port
        return kargs
    else:
        return kargs