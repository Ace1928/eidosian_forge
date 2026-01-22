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
class FileExtensionValidator:
    message = _('File extension “%(extension)s” is not allowed. Allowed extensions are: %(allowed_extensions)s.')
    code = 'invalid_extension'

    def __init__(self, allowed_extensions=None, message=None, code=None):
        if allowed_extensions is not None:
            allowed_extensions = [allowed_extension.lower() for allowed_extension in allowed_extensions]
        self.allowed_extensions = allowed_extensions
        if message is not None:
            self.message = message
        if code is not None:
            self.code = code

    def __call__(self, value):
        extension = Path(value.name).suffix[1:].lower()
        if self.allowed_extensions is not None and extension not in self.allowed_extensions:
            raise ValidationError(self.message, code=self.code, params={'extension': extension, 'allowed_extensions': ', '.join(self.allowed_extensions), 'value': value})

    def __eq__(self, other):
        return isinstance(other, self.__class__) and self.allowed_extensions == other.allowed_extensions and (self.message == other.message) and (self.code == other.code)