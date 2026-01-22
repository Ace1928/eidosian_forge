from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import abc
import re
import unicodedata
from googlecloudsdk.core import log
from googlecloudsdk.core.resource import resource_exceptions
from googlecloudsdk.core.resource import resource_lex
from googlecloudsdk.core.resource import resource_property
from googlecloudsdk.core.util import encoding
from googlecloudsdk.core.util import times
import six
def _MatchOneWordInText(backend, key, op, warned_attribute, value, pattern):
    """Returns True if value word matches pattern.

  Args:
    backend: The parser backend object.
    key: The parsed expression key.
    op: The expression operator string.
    warned_attribute: Deprecation warning Boolean attribute name.
    value: The value to be matched by pattern.
    pattern: An (operand, standard_regex, deprecated_regex) tuple.

  Raises:
    ValueError: To catch codebase reliance on deprecated usage.

  Returns:
    True if pattern matches value.

  Examples:
    See surface/topic/filters.py for a table of example matches.
  """
    operand, standard_regex, deprecated_regex = pattern
    if isinstance(value, float):
        try:
            if value == float(operand):
                return True
        except ValueError:
            pass
        if value == 0 and operand.lower() == 'false':
            return True
        if value == 1 and operand.lower() == 'true':
            return True
        text = re.sub('\\.0*$', '', _Stringize(value))
    elif value == operand:
        return True
    elif value is None:
        if operand in ('', None):
            return True
        if operand == '*' and op == ':':
            return False
        text = 'null'
    else:
        text = NormalizeForSearch(value, html=True)
    matched = bool(standard_regex.search(text))
    if not deprecated_regex:
        return matched
    deprecated_matched = bool(deprecated_regex.search(text))
    if len(key) == 1 and key[0] in ['zone', 'region']:
        deprecated_matched |= bool(deprecated_regex.search(text.split('/')[-1]))
    if matched != deprecated_matched and warned_attribute and (not getattr(backend, warned_attribute, False)):
        setattr(backend, warned_attribute, True)
        old_match = 'matches' if deprecated_matched else 'does not match'
        new_match = 'will match' if matched else 'will not match'
        log.warning('--filter : operator evaluation is changing for consistency across Google APIs.  {key}{op}{operand} currently {old_match} but {new_match} in the near future.  Run `gcloud topic filters` for details.'.format(key=resource_lex.GetKeyName(key), op=op, operand=operand, old_match=old_match, new_match=new_match))
    return deprecated_matched