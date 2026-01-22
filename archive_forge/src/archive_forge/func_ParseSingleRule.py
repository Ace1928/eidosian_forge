from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.api_lib.bigtable import util
from googlecloudsdk.calliope import base
from googlecloudsdk.calliope import exceptions
from googlecloudsdk.core.util import times
def ParseSingleRule(rule):
    """Parses GC rules from a rule string.

  Args:
    rule: A string representing a GC rule, e.g. `maxage=10d`

  Returns:
    A GcRule object.

  Raises:
    BadArgumentExpection: the input is mal-formatted.
  """
    rule_parts = rule.split('=')
    if len(rule_parts) != 2 or not rule_parts[1]:
        raise exceptions.BadArgumentException('--column-families', 'Invalid union or intersection rule: {0}'.format(rule))
    if rule_parts[0] == 'maxage':
        return util.GetAdminMessages().GcRule(maxAge=ConvertDurationToSeconds(rule_parts[1]))
    elif rule_parts[0] == 'maxversions':
        return util.GetAdminMessages().GcRule(maxNumVersions=int(rule_parts[1]))
    else:
        raise exceptions.BadArgumentException('--column-families', 'Invalid union or intersection rule: {0}'.format(rule))