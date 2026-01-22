from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import collections
import enum
import re
from apitools.base.protorpclite import messages
from apitools.base.py import encoding
from googlecloudsdk.calliope import arg_parsers
from googlecloudsdk.calliope import base
from googlecloudsdk.calliope.concepts import util as format_util
from googlecloudsdk.core import properties
from googlecloudsdk.core.resource import resource_property
from googlecloudsdk.core.util import http_encoding
import six
def _ValidateAndParseMappings(self):
    """Validates and parses choice to enum mappings.

    Validates and parses choice to enum mappings including any custom mappings.

    Raises:
      ValueError: custom_mappings does not contain correct number of mapped
        values.
      TypeError: custom_mappings is incorrect type or contains incorrect types
        for mapped values.
    """
    if self._custom_mappings:
        if not isinstance(self._custom_mappings, dict):
            raise TypeError(self._CUSTOM_MAPPING_ERROR.format(self._custom_mappings))
        enum_strings = set([x.name for x in self._enum])
        diff = set(self._custom_mappings.keys()) - enum_strings
        if diff:
            raise ValueError('custom_mappings [{}] may only contain mappings for enum values. invalid values:[{}]'.format(', '.join(self._custom_mappings.keys()), ', '.join(diff)))
        try:
            self._ParseCustomMappingsFromTuples()
        except (TypeError, ValueError):
            self._ParseCustomMappingsFromStrings()
    else:
        if callable(self._filter):
            self._filtered_enum = [e for e in self._enum if self._filter(e.name)]
        self._choice_to_enum = {EnumNameToChoice(x.name): x for x in self._filtered_enum}
        self._enum_to_choice = {y.name: x for x, y in six.iteritems(self._choice_to_enum)}
        self._choices = sorted(self._choice_to_enum.keys())