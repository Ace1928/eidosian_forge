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
class SemVer(TypeWithIntervalConstraint):
    """SemVer concept."""

    def BuildHelpText(self):
        """Appends SemVer syntax to the original help text."""
        return '{}Must be a string representing a SemVer number of the form _MAJOR_._MINOR_._PATCH_, where omitted trailing parts default to 0. {}See https://semver.org/ for more information.'.format(_Insert(super(SemVer, self).BuildHelpText()), _Insert(self.Constraints()))

    def Convert(self, string):
        """Converts a SemVer object from string returns it."""
        if not string:
            return None
        try:
            parts = string.split('.')
            while len(parts) < 3:
                parts.append('0')
            string = '.'.join(parts)
            return semver.SemVer(string)
        except semver.ParseError as e:
            raise exceptions.ParseError(self.GetPresentationName(), _SubException(e))

    def Display(self, value):
        """Returns the display string for a SemVer object value."""
        return '{}.{}.{}'.format(value.major, value.minor, value.patch)