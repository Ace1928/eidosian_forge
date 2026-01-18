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
@cached_property
def url_patterns(self):
    patterns = getattr(self.urlconf_module, 'urlpatterns', self.urlconf_module)
    try:
        iter(patterns)
    except TypeError as e:
        msg = "The included URLconf '{name}' does not appear to have any patterns in it. If you see the 'urlpatterns' variable with valid patterns in the file then the issue is probably caused by a circular import."
        raise ImproperlyConfigured(msg.format(name=self.urlconf_name)) from e
    return patterns