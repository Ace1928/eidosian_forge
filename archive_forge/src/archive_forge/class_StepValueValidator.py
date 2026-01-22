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
class StepValueValidator(BaseValidator):
    message = _('Ensure this value is a multiple of step size %(limit_value)s.')
    code = 'step_size'

    def __init__(self, limit_value, message=None, offset=None):
        super().__init__(limit_value, message)
        if offset is not None:
            self.message = _('Ensure this value is a multiple of step size %(limit_value)s, starting from %(offset)s, e.g. %(offset)s, %(valid_value1)s, %(valid_value2)s, and so on.')
        self.offset = offset

    def __call__(self, value):
        if self.offset is None:
            super().__call__(value)
        else:
            cleaned = self.clean(value)
            limit_value = self.limit_value() if callable(self.limit_value) else self.limit_value
            if self.compare(cleaned, limit_value):
                offset = cleaned.__class__(self.offset)
                params = {'limit_value': limit_value, 'offset': offset, 'valid_value1': offset + limit_value, 'valid_value2': offset + 2 * limit_value}
                raise ValidationError(self.message, code=self.code, params=params)

    def compare(self, a, b):
        offset = 0 if self.offset is None else self.offset
        return not math.isclose(math.remainder(a - offset, b), 0, abs_tol=1e-09)