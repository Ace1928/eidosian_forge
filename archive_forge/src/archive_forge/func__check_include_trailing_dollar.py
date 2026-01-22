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
def _check_include_trailing_dollar(self):
    regex_pattern = self.regex.pattern
    if regex_pattern.endswith('$') and (not regex_pattern.endswith('\\$')):
        return [Warning("Your URL pattern {} uses include with a route ending with a '$'. Remove the dollar from the route to avoid problems including URLs.".format(self.describe()), id='urls.W001')]
    else:
        return []