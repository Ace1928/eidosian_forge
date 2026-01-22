from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import copy
import re
from googlecloudsdk.core.resource import resource_exceptions
from googlecloudsdk.core.resource import resource_filter
from googlecloudsdk.core.resource import resource_lex
from googlecloudsdk.core.resource import resource_projection_spec
from googlecloudsdk.core.resource import resource_transform
import six
def _ParseAttributes(self):
    """Parses a comma separated [no-]name[=value] projection attribute list.

    The initial '[' has already been consumed by the caller.

    Raises:
      ExpressionSyntaxError: The expression has a syntax error.
    """
    while True:
        name = self._lex.Token('=,])', space=False)
        if name:
            if self._lex.IsCharacter('='):
                value = self._lex.Token(',])', space=False, convert=True)
            else:
                value = 1
            if isinstance(value, six.string_types):
                value = value.replace('\\n', '\n').replace('\\t', '\t')
            self._projection.AddAttribute(name, value)
            if name.startswith('no-'):
                self._projection.DelAttribute(name[3:])
            else:
                self._projection.DelAttribute('no-' + name)
        if self._lex.IsCharacter(']'):
            break
        if not self._lex.IsCharacter(','):
            raise resource_exceptions.ExpressionSyntaxError('Expected ] in attribute list [{0}].'.format(self._lex.Annotate()))