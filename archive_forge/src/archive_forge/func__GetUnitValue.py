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
def _GetUnitValue(self, kind, unit):
    """Returns the integer unit suffix and value for unit."""
    if self._type_abbr:
        unit = scaled_integer.DeleteTypeAbbr(unit)
    try:
        return (unit, scaled_integer.GetUnitSize(unit))
    except ValueError as e:
        raise exceptions.ConstraintError(self.name, kind, unit, _SubException(e) + '.')