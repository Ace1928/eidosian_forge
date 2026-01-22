import ipaddress
import math
import re
from pathlib import Path
from urllib.parse import urlsplit, urlunsplit
from django.core.exceptions import ValidationError
from django.utils.deconstruct import deconstructible
from django.utils.encoding import punycode
from django.utils.ipv6 import is_valid_ipv6_address
from django.utils.regex_helper import _lazy_re_compile
from django.utils.translation import gettext_lazy as _
from django.utils.translation import ngettext_lazy
@deconstructible
class URLValidator(RegexValidator):
    ul = '¡-\uffff'
    ipv4_re = '(?:0|25[0-5]|2[0-4][0-9]|1[0-9]?[0-9]?|[1-9][0-9]?)(?:\\.(?:0|25[0-5]|2[0-4][0-9]|1[0-9]?[0-9]?|[1-9][0-9]?)){3}'
    ipv6_re = '\\[[0-9a-f:.]+\\]'
    hostname_re = '[a-z' + ul + '0-9](?:[a-z' + ul + '0-9-]{0,61}[a-z' + ul + '0-9])?'
    domain_re = '(?:\\.(?!-)[a-z' + ul + '0-9-]{1,63}(?<!-))*'
    tld_re = '\\.(?!-)(?:[a-z' + ul + '-]{2,63}|xn--[a-z0-9]{1,59})(?<!-)\\.?'
    host_re = '(' + hostname_re + domain_re + tld_re + '|localhost)'
    regex = _lazy_re_compile('^(?:[a-z0-9.+-]*)://(?:[^\\s:@/]+(?::[^\\s:@/]*)?@)?(?:' + ipv4_re + '|' + ipv6_re + '|' + host_re + ')(?::[0-9]{1,5})?(?:[/?#][^\\s]*)?\\Z', re.IGNORECASE)
    message = _('Enter a valid URL.')
    schemes = ['http', 'https', 'ftp', 'ftps']
    unsafe_chars = frozenset('\t\r\n')
    max_length = 2048

    def __init__(self, schemes=None, **kwargs):
        super().__init__(**kwargs)
        if schemes is not None:
            self.schemes = schemes

    def __call__(self, value):
        if not isinstance(value, str) or len(value) > self.max_length:
            raise ValidationError(self.message, code=self.code, params={'value': value})
        if self.unsafe_chars.intersection(value):
            raise ValidationError(self.message, code=self.code, params={'value': value})
        scheme = value.split('://')[0].lower()
        if scheme not in self.schemes:
            raise ValidationError(self.message, code=self.code, params={'value': value})
        try:
            splitted_url = urlsplit(value)
        except ValueError:
            raise ValidationError(self.message, code=self.code, params={'value': value})
        try:
            super().__call__(value)
        except ValidationError as e:
            if value:
                scheme, netloc, path, query, fragment = splitted_url
                try:
                    netloc = punycode(netloc)
                except UnicodeError:
                    raise e
                url = urlunsplit((scheme, netloc, path, query, fragment))
                super().__call__(url)
            else:
                raise
        else:
            host_match = re.search('^\\[(.+)\\](?::[0-9]{1,5})?$', splitted_url.netloc)
            if host_match:
                potential_ip = host_match[1]
                try:
                    validate_ipv6_address(potential_ip)
                except ValidationError:
                    raise ValidationError(self.message, code=self.code, params={'value': value})
        if splitted_url.hostname is None or len(splitted_url.hostname) > 253:
            raise ValidationError(self.message, code=self.code, params={'value': value})