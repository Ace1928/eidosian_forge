from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import re
from googlecloudsdk.command_lib.scc.errors import InvalidSCCInputError
from googlecloudsdk.command_lib.scc.hooks import CleanUpUserInput
def _NormalizeOrganization(organization):
    """Validates an organization name or id."""
    if '/' in organization:
        if _ORGANIZATION_NAME_PATTERN.fullmatch(organization):
            return organization + _PARENT_SUFFIX
        raise _InvalidFullResourcePathForPattern(_ORGANIZATION_NAME_PATTERN)
    if _ORGANIZATION_ID_PATTERN.fullmatch(organization):
        return 'organizations/' + organization + _PARENT_SUFFIX
    raise InvalidSCCInputError("Organization does not match the pattern '%s'." % _ORGANIZATION_ID_PATTERN.pattern)