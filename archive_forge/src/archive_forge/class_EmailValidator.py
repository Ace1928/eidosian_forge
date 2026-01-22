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
class EmailValidator:
    message = _('Enter a valid email address.')
    code = 'invalid'
    user_regex = _lazy_re_compile('(^[-!#$%&\'*+/=?^_`{}|~0-9A-Z]+(\\.[-!#$%&\'*+/=?^_`{}|~0-9A-Z]+)*\\Z|^"([\\001-\\010\\013\\014\\016-\\037!#-\\[\\]-\\177]|\\\\[\\001-\\011\\013\\014\\016-\\177])*"\\Z)', re.IGNORECASE)
    domain_regex = _lazy_re_compile('((?:[A-Z0-9](?:[A-Z0-9-]{0,61}[A-Z0-9])?\\.)+)(?:[A-Z0-9-]{2,63}(?<!-))\\Z', re.IGNORECASE)
    literal_regex = _lazy_re_compile('\\[([A-F0-9:.]+)\\]\\Z', re.IGNORECASE)
    domain_allowlist = ['localhost']

    def __init__(self, message=None, code=None, allowlist=None):
        if message is not None:
            self.message = message
        if code is not None:
            self.code = code
        if allowlist is not None:
            self.domain_allowlist = allowlist

    def __call__(self, value):
        if not value or '@' not in value or len(value) > 320:
            raise ValidationError(self.message, code=self.code, params={'value': value})
        user_part, domain_part = value.rsplit('@', 1)
        if not self.user_regex.match(user_part):
            raise ValidationError(self.message, code=self.code, params={'value': value})
        if domain_part not in self.domain_allowlist and (not self.validate_domain_part(domain_part)):
            try:
                domain_part = punycode(domain_part)
            except UnicodeError:
                pass
            else:
                if self.validate_domain_part(domain_part):
                    return
            raise ValidationError(self.message, code=self.code, params={'value': value})

    def validate_domain_part(self, domain_part):
        if self.domain_regex.match(domain_part):
            return True
        literal_match = self.literal_regex.match(domain_part)
        if literal_match:
            ip_address = literal_match[1]
            try:
                validate_ipv46_address(ip_address)
                return True
            except ValidationError:
                pass
        return False

    def __eq__(self, other):
        return isinstance(other, EmailValidator) and self.domain_allowlist == other.domain_allowlist and (self.message == other.message) and (self.code == other.code)