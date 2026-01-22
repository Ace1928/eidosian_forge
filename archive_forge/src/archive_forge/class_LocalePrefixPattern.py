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
class LocalePrefixPattern:

    def __init__(self, prefix_default_language=True):
        self.prefix_default_language = prefix_default_language
        self.converters = {}

    @property
    def regex(self):
        return re.compile(re.escape(self.language_prefix))

    @property
    def language_prefix(self):
        language_code = get_language() or settings.LANGUAGE_CODE
        if language_code == settings.LANGUAGE_CODE and (not self.prefix_default_language):
            return ''
        else:
            return '%s/' % language_code

    def match(self, path):
        language_prefix = self.language_prefix
        if path.startswith(language_prefix):
            return (path.removeprefix(language_prefix), (), {})
        return None

    def check(self):
        return []

    def describe(self):
        return "'{}'".format(self)

    def __str__(self):
        return self.language_prefix