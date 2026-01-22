import functools
import inspect
import re
import string
from importlib import import_module
from pickle import PicklingError
from urllib.parse import quote
from asgiref.local import Local
from django.conf import settings
from django.core.checks import Error, Warning
from django.core.checks.urls import check_resolver
from django.core.exceptions import ImproperlyConfigured, ViewDoesNotExist
from django.utils.datastructures import MultiValueDict
from django.utils.functional import cached_property
from django.utils.http import RFC3986_SUBDELIMS, escape_leading_slashes
from django.utils.regex_helper import _lazy_re_compile, normalize
from django.utils.translation import get_language
from .converters import get_converter
from .exceptions import NoReverseMatch, Resolver404
from .utils import get_callable
def _check_pattern_unmatched_angle_brackets(self):
    warnings = []
    msg = "Your URL pattern %s has an unmatched '%s' bracket."
    brackets = re.findall('[<>]', str(self._route))
    open_bracket_counter = 0
    for bracket in brackets:
        if bracket == '<':
            open_bracket_counter += 1
        elif bracket == '>':
            open_bracket_counter -= 1
            if open_bracket_counter < 0:
                warnings.append(Warning(msg % (self.describe(), '>'), id='urls.W010'))
                open_bracket_counter = 0
    if open_bracket_counter > 0:
        warnings.append(Warning(msg % (self.describe(), '<'), id='urls.W010'))
    return warnings