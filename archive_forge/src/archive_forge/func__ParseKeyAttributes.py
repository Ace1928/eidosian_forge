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
def _ParseKeyAttributes(self, key, attribute):
    """Parses one or more key attributes and adds them to attribute.

    The initial ':' has been consumed by the caller.

    Args:
      key: The parsed key name of the attributes.
      attribute: Add the parsed transform to this resource_projector._Attribute.

    Raises:
      ExpressionSyntaxError: The expression has a syntax error.
    """
    while True:
        name = self._lex.Token('=:,)', space=False)
        here = self._lex.GetPosition()
        if self._lex.IsCharacter('=', eoi_ok=True):
            boolean_value = False
            value = self._lex.Token(':,)', space=False, convert=True)
        else:
            boolean_value = True
            if name.startswith('no-'):
                name = name[3:]
                value = False
            else:
                value = True
        if name in self._BOOLEAN_ATTRIBUTES:
            if not boolean_value:
                raise resource_exceptions.ExpressionSyntaxError('value not expected [{0}].'.format(self._lex.Annotate(here)))
        elif boolean_value and name not in self._OPTIONAL_BOOLEAN_ATTRIBUTES:
            raise resource_exceptions.ExpressionSyntaxError('value expected [{0}].'.format(self._lex.Annotate(here)))
        if name == 'alias':
            if not value:
                raise resource_exceptions.ExpressionSyntaxError('Cannot unset alias [{0}].'.format(self._lex.Annotate(here)))
            self._projection.AddAlias(value, key, attribute)
        elif name == 'align':
            if value not in resource_projection_spec.ALIGNMENTS:
                raise resource_exceptions.ExpressionSyntaxError('Unknown alignment [{0}].'.format(self._lex.Annotate(here)))
            attribute.align = value
        elif name == 'format':
            attribute.subformat = value or ''
        elif name == 'label':
            attribute.label = value or ''
        elif name == 'optional':
            attribute.optional = value
        elif name == 'reverse':
            attribute.reverse = value
        elif name == 'sort':
            attribute.order = value
        elif name == 'width':
            attribute.width = value
        elif name == 'wrap':
            attribute.wrap = value
        else:
            raise resource_exceptions.ExpressionSyntaxError('Unknown key attribute [{0}].'.format(self._lex.Annotate(here)))
        if not self._lex.IsCharacter(':'):
            break