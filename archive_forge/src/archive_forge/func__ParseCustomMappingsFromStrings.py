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
def _ParseCustomMappingsFromStrings(self):
    """Parses choice to enum mappings from custom_mapping with strings.

     Parses choice mappings from dict mapping Enum strings to choice
     values {str -> str} mapping.

    Raises:
      TypeError - Custom choices are not strings
    """
    self._choice_to_enum = {}
    self._choices = []
    for enum_string, choice_string in sorted(six.iteritems(self._custom_mappings)):
        if not isinstance(choice_string, six.string_types):
            raise TypeError(self._CUSTOM_MAPPING_ERROR.format(self._custom_mappings))
        self._choice_to_enum[choice_string] = self._enum(enum_string)
        self._choices.append(choice_string)
    self._enum_to_choice = self._custom_mappings