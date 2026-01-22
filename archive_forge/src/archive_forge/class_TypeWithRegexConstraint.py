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
class TypeWithRegexConstraint(ConceptType):
    """Concept type with regex constraint.

  Attributes:
    _regex: string, an unanchored regular expression pattern that must match
      valid values.
    _constraint_details: string, optional prose that describes the regex
      constraint.
  """

    def __init__(self, name, regex=None, constraint_details=None, **kwargs):
        super(TypeWithRegexConstraint, self).__init__(name, **kwargs)
        self._regex = regex
        self._constraint_details = constraint_details

    def Constraints(self):
        """Returns the type constraints message text if any."""
        if not self._regex:
            return ''
        if self._constraint_details:
            return self._constraint_details
        return 'The value must match the regular expression ```{}```.'.format(self._regex)

    def Validate(self, value):
        if self._regex and (not re.match(self._regex, self.Display(value))):
            raise exceptions.ValidationError(self.GetPresentationName(), 'Value [{}] does not match [{}].'.format(self.Display(value), self._regex))