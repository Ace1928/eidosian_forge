from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import json
import re
import six
from six.moves import range  # pylint: disable=redefined-builtin
def EvaluateGlobalRestriction(resource, restriction, pattern):
    """Returns True if any attribute value in resource matches the RE pattern.

  This function is called to evaluate a global restriction on a resource. For
  example, --filter="Foo.Bar" results in a call like this on each resource item:

    resource_property.EvaluateGlobalRestriction(
      resource,
      'Foo.Bar',
      re.compile(re.escape('Foo.Bar'), re.IGNORECASE),
    )

  Args:
    resource: The object to check.
    restriction: The global restriction string.
    pattern: The global restriction pattern for matcing resource values.

  Returns:
    True if any attribute value in resource matches the RE pattern.
  """
    if not resource:
        return False
    if isinstance(resource, six.string_types):
        try:
            return bool(pattern.search(resource))
        except TypeError:
            pass
    if isinstance(resource, (float, int)):
        try:
            return bool(pattern.search(str(resource)))
        except TypeError:
            pass
    try:
        for key, value in six.iteritems(resource):
            if not key.startswith('_') and EvaluateGlobalRestriction(value, restriction, pattern):
                return True
    except AttributeError:
        try:
            for value in resource:
                if EvaluateGlobalRestriction(value, restriction, pattern):
                    return True
            return False
        except TypeError:
            pass
    try:
        for key, value in six.iteritems(resource.__dict__):
            if not key.startswith('_') and EvaluateGlobalRestriction(value, restriction, pattern):
                return True
    except AttributeError:
        pass
    return False