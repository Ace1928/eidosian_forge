from googlecloudsdk.command_lib.concepts import concept_managers
from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import copy
import io
import re
from googlecloudsdk.calliope.concepts import deps as deps_lib
from googlecloudsdk.command_lib.concepts import base
from googlecloudsdk.command_lib.concepts import dependency_managers
from googlecloudsdk.command_lib.concepts import exceptions
from googlecloudsdk.command_lib.concepts import names
from googlecloudsdk.core.util import scaled_integer
from googlecloudsdk.core.util import semver
from googlecloudsdk.core.util import times
import six
class ScaledInteger(TypeWithIntervalConstraint):
    """ISO Decimal/Binary scaled Integer value concept.

  ISO/IEC prefixes: 1k == 1000, 1ki == 1024.

  Attributes:
    _default_unit: string, the unit suffix if none is specified.
    _output_unit: string, the implicit output unit. Integer values are
      divided by the output unit value.
    _output_unit_value: int, the output unit value.
    _type_abbr: string, the type abbreviation, for example 'b/s' or 'Hz'.
    _type_details: string, prose that describes type syntax details.
  """

    def __init__(self, name, default_unit=None, output_unit=None, type_abbr='B', type_details=None, **kwargs):
        self.name = name
        self._type_abbr = type_abbr
        self._default_unit = default_unit
        if self._default_unit:
            self._default_unit, _ = self._GetUnitValue('default scaled integer unit', self._default_unit)
        self._output_unit = output_unit
        if self._output_unit:
            self._output_unit, self._output_unit_value = self._GetUnitValue('output scaled integer unit', self._output_unit)
        else:
            self._output_unit_value = 0
        self._type_details = type_details or 'Must be a string representing an ISO/IEC Decimal/Binary scaled integer. For example, 1k == 1000 and 1ki == 1024. '
        super(ScaledInteger, self).__init__(name, **kwargs)

    def _GetUnitValue(self, kind, unit):
        """Returns the integer unit suffix and value for unit."""
        if self._type_abbr:
            unit = scaled_integer.DeleteTypeAbbr(unit)
        try:
            return (unit, scaled_integer.GetUnitSize(unit))
        except ValueError as e:
            raise exceptions.ConstraintError(self.name, kind, unit, _SubException(e) + '.')

    def BuildHelpText(self):
        """Appends ISO Decimal/Binary scaled integer syntax to the help text."""
        if self._default_unit:
            default_unit = 'The default unit is `{}`. '.format(self._default_unit)
        else:
            default_unit = ''
        if self._output_unit:
            output_unit = 'The output unit is `{}`. Integer values are divided by the unit value. '.format(self._output_unit)
        else:
            output_unit = ''
        if self._type_abbr:
            type_abbr = 'The default type abbreviation is `{}`. '.format(self._type_abbr)
        else:
            type_abbr = ''
        return '{}{}{}{}{}{}See https://en.wikipedia.org/wiki/Binary_prefix for details.'.format(_Insert(super(ScaledInteger, self).BuildHelpText()), self._type_details, default_unit, output_unit, type_abbr, _Insert(self.Constraints()))

    def Convert(self, string):
        if not string:
            return None
        try:
            value = scaled_integer.ParseInteger(string, default_unit=self._default_unit, type_abbr=self._type_abbr)
            if self._output_unit_value:
                value //= self._output_unit_value
            return value
        except ValueError as e:
            raise exceptions.ParseError(self.GetPresentationName(), 'Failed to parse binary/decimal scaled integer [{}]: {}.'.format(string, _SubException(e)))

    def Display(self, value):
        """Returns the display string for a binary scaled value."""
        if self._output_unit_value:
            value *= self._output_unit_value
        return scaled_integer.FormatInteger(value, type_abbr=self._type_abbr)