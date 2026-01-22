from __future__ import print_function
from __future__ import unicode_literals
import collections
import contextlib
import gzip
import json
import keyword
import logging
import os
import re
import tempfile
import six
from six.moves import urllib_parse
import six.moves.urllib.error as urllib_error
import six.moves.urllib.request as urllib_request
class Names(object):
    """Utility class for cleaning and normalizing names in a fixed style."""
    DEFAULT_NAME_CONVENTION = 'LOWER_CAMEL'
    NAME_CONVENTIONS = ['LOWER_CAMEL', 'LOWER_WITH_UNDER', 'NONE']

    def __init__(self, strip_prefixes, name_convention=None, capitalize_enums=False):
        self.__strip_prefixes = sorted(strip_prefixes, key=_SortLengthFirstKey)
        self.__name_convention = name_convention or self.DEFAULT_NAME_CONVENTION
        self.__capitalize_enums = capitalize_enums

    @staticmethod
    def __FromCamel(name, separator='_'):
        name = re.sub('([a-z0-9])([A-Z])', '\\1%s\\2' % separator, name)
        return name.lower()

    @staticmethod
    def __ToCamel(name, separator='_'):
        return ''.join((s[0:1].upper() + s[1:] for s in name.split(separator)))

    @staticmethod
    def __ToLowerCamel(name, separator='_'):
        name = Names.__ToCamel(name, separator=separator)
        return name[0].lower() + name[1:]

    def __StripName(self, name):
        """Strip strip_prefix entries from name."""
        if not name:
            return name
        for prefix in self.__strip_prefixes:
            if name.startswith(prefix):
                return name[len(prefix):]
        return name

    @staticmethod
    def CleanName(name):
        """Perform generic name cleaning."""
        name = re.sub('[^_A-Za-z0-9]', '_', name)
        if name[0].isdigit():
            name = '_%s' % name
        while keyword.iskeyword(name) or name == 'exec':
            name = '%s_' % name
        if name.startswith('__'):
            name = 'f%s' % name
        return name

    @staticmethod
    def NormalizeRelativePath(path):
        """Normalize camelCase entries in path."""
        path_components = path.split('/')
        normalized_components = []
        for component in path_components:
            if re.match('{[A-Za-z0-9_]+}$', component):
                normalized_components.append('{%s}' % Names.CleanName(component[1:-1]))
            else:
                normalized_components.append(component)
        return '/'.join(normalized_components)

    def NormalizeEnumName(self, enum_name):
        if self.__capitalize_enums:
            enum_name = enum_name.upper()
        return self.CleanName(enum_name)

    def ClassName(self, name, separator='_'):
        """Generate a valid class name from name."""
        if name is None:
            return name
        if name.startswith(('protorpc.', 'message_types.', 'apitools.base.protorpclite.', 'apitools.base.protorpclite.message_types.')):
            return name
        name = self.__StripName(name)
        name = self.__ToCamel(name, separator=separator)
        return self.CleanName(name)

    def MethodName(self, name, separator='_'):
        """Generate a valid method name from name."""
        if name is None:
            return None
        name = Names.__ToCamel(name, separator=separator)
        return Names.CleanName(name)

    def FieldName(self, name):
        """Generate a valid field name from name."""
        name = self.__StripName(name)
        if self.__name_convention == 'LOWER_CAMEL':
            name = Names.__ToLowerCamel(name)
        elif self.__name_convention == 'LOWER_WITH_UNDER':
            name = Names.__FromCamel(name)
        return Names.CleanName(name)