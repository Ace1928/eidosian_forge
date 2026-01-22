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
class RoutePattern(CheckURLMixin):
    regex = LocaleRegexDescriptor('_route')

    def __init__(self, route, name=None, is_endpoint=False):
        self._route = route
        self._regex_dict = {}
        self._is_endpoint = is_endpoint
        self.name = name
        self.converters = _route_to_regex(str(route), is_endpoint)[1]

    def match(self, path):
        match = self.regex.search(path)
        if match:
            kwargs = match.groupdict()
            for key, value in kwargs.items():
                converter = self.converters[key]
                try:
                    kwargs[key] = converter.to_python(value)
                except ValueError:
                    return None
            return (path[match.end():], (), kwargs)
        return None

    def check(self):
        warnings = [*self._check_pattern_startswith_slash(), *self._check_pattern_unmatched_angle_brackets()]
        route = self._route
        if '(?P<' in route or route.startswith('^') or route.endswith('$'):
            warnings.append(Warning("Your URL pattern {} has a route that contains '(?P<', begins with a '^', or ends with a '$'. This was likely an oversight when migrating to django.urls.path().".format(self.describe()), id='2_0.W001'))
        return warnings

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

    def _compile(self, route):
        return re.compile(_route_to_regex(route, self._is_endpoint)[0])

    def __str__(self):
        return str(self._route)