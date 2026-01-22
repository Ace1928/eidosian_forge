from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import re
from googlecloudsdk.core.resource import resource_exceptions
from googlecloudsdk.core.resource import resource_expr
from googlecloudsdk.core.resource import resource_lex
from googlecloudsdk.core.resource import resource_projection_spec
from googlecloudsdk.core.resource import resource_property
import six
def GetAllKeys(expression):
    """Recursively collects all keys in compiled filter expression."""
    keys = set()
    if expression.contains_key:
        keys.add(tuple(expression.key))
    for _, obj in six.iteritems(vars(expression)):
        if hasattr(obj, 'contains_key'):
            keys |= GetAllKeys(obj)
    return keys