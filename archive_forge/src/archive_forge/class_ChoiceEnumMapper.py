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
class ChoiceEnumMapper(object):
    """Utility class for mapping apitools Enum messages to argparse choice args.

  Dynamically builds a base.Argument from an enum message.
  Derives choice values from supplied enum or an optional custom_mapping dict
  (see below).

  Class Attributes:
   choices: Either a list of strings [str] specifying the commandline choice
       values or an ordered dict of choice value to choice help string mappings
       {str -> str}
   enum: underlying enum whos values map to supplied choices.
   choice_arg: base.Argument object
   choice_mappings: Mapping of argparse choice value strings to enum values.
   custom_mappings: Optional dict mapping enum values to a custom
     argparse choice value. To maintain compatiblity with base.ChoiceAgrument(),
     dict can be either:
     {str-> str} - Enum String value to choice argument value i.e.
     {'MY_MUCH_LONGER_ENUM_VALUE':'short-arg'}
     OR
     {str -> (str, str)} -  Enum string value to  tuple of
     (choice argument value, choice help string) i.e.
     {'MY_MUCH_LONGER_ENUM_VALUE':('short-arg','My short arg help text.')}
  """
    _CUSTOM_MAPPING_ERROR = 'custom_mappings must be a dict of enum string values to argparse argument choices. Choices must be either a string or a string tuple of (choice, choice_help_text): [{}]'

    def __init__(self, arg_name, message_enum, custom_mappings=None, help_str=None, required=False, action=None, metavar=None, dest=None, default=None, hidden=False, include_filter=None):
        """Initialize ChoiceEnumMapper.

    Args:
      arg_name: str, The name of the argparse argument to create
      message_enum: apitools.Enum, the enum to map
      custom_mappings: See Above.
      help_str: string, pass through for base.Argument,
        see base.ChoiceArgument().
      required: boolean,string, pass through for base.Argument,
          see base.ChoiceArgument().
      action: string or argparse.Action, string, pass through for base.Argument,
          see base.ChoiceArgument().
      metavar: string,  string, pass through for base.Argument,
          see base.ChoiceArgument()..
      dest: string, string, pass through for base.Argument,
          see base.ChoiceArgument().
      default: string, string, pass through for base.Argument,
          see base.ChoiceArgument().
      hidden: boolean, pass through for base.Argument,
          see base.ChoiceArgument().
      include_filter: callable, function of type string->bool used to filter
          enum values from message_enum that should be included in choices.
          If include_filter returns True for a particular enum value, it will be
          included otherwise it will be excluded. This is ignored if
          custom_mappings is specified.

    Raises:
      ValueError: If no enum is given, mappings are incomplete
      TypeError: If invalid values are passed for base.Argument or
       custom_mapping
    """
        if not isinstance(message_enum, messages._EnumClass):
            raise ValueError('Invalid Message Enum: [{}]'.format(message_enum))
        self._arg_name = arg_name
        self._enum = message_enum
        self._custom_mappings = custom_mappings
        if include_filter is not None and (not callable(include_filter)):
            raise TypeError('include_filter must be callable received [{}]'.format(include_filter))
        self._filter = include_filter
        self._filtered_enum = self._enum
        self._ValidateAndParseMappings()
        self._choice_arg = base.ChoiceArgument(arg_name, self.choices, help_str=help_str, required=required, action=action, metavar=metavar, dest=dest, default=default, hidden=hidden)

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

    def _ParseCustomMappingsFromTuples(self):
        """Parses choice to enum mappings from custom_mapping with tuples.

     Parses choice mappings from dict mapping Enum strings to a tuple of
     choice values and choice help {str -> (str, str)} mapping.

    Raises:
      TypeError - Custom choices are not not valid (str,str) tuples.
    """
        self._choice_to_enum = {}
        self._enum_to_choice = {}
        self._choices = collections.OrderedDict()
        for enum_string, (choice, help_str) in sorted(six.iteritems(self._custom_mappings)):
            self._choice_to_enum[choice] = self._enum(enum_string)
            self._enum_to_choice[enum_string] = choice
            self._choices[choice] = help_str

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

    def GetChoiceForEnum(self, enum_value):
        """Converts an enum value to a choice argument value."""
        return self._enum_to_choice.get(six.text_type(enum_value))

    def GetEnumForChoice(self, choice_value):
        """Converts a mapped string choice value to an enum."""
        return self._choice_to_enum.get(choice_value)

    @property
    def choices(self):
        return self._choices

    @property
    def enum(self):
        return self._enum

    @property
    def filtered_enum(self):
        return self._filtered_enum

    @property
    def choice_arg(self):
        return self._choice_arg

    @property
    def choice_mappings(self):
        return self._choice_to_enum

    @property
    def custom_mappings(self):
        return self._custom_mappings

    @property
    def include_filter(self):
        return self._filter