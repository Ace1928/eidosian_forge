from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import copy
import re
from googlecloudsdk.core.resource import resource_exceptions
from googlecloudsdk.core.resource import resource_projection_spec
from googlecloudsdk.core.resource import resource_property
from googlecloudsdk.core.resource import resource_transform
import six
from six.moves import map  # pylint: disable=redefined-builtin
from six.moves import range  # pylint: disable=redefined-builtin
def KeyWithAttribute(self):
    """Parses a resource key from the expression.

    A resource key is a '.' separated list of names with optional [] slice or
    [NUMBER] array indices. Names containing _RESERVED_OPERATOR_CHARS must be
    quoted. For example, "k.e.y".value has two name components, 'k.e.y' and
    'value'.

    A parsed key is encoded as an ordered list of tokens, where each token may
    be:

      KEY VALUE   PARSED VALUE  DESCRIPTION
      ---------   ------------  -----------
      name        string        A dotted name list element.
      [NUMBER]    NUMBER        An array index.
      []          None          An array slice.

    For example, the key 'abc.def[123].ghi[].jkl' parses to this encoded list:
      ['abc', 'def', 123, 'ghi', None, 'jkl']

    Raises:
      ExpressionKeyError: The expression has a key syntax error.

    Returns:
      (key, attribute) The parsed key and attribute. attribute is the alias
        attribute if there was an alias expansion, None otherwise.
    """
    key = []
    attribute = None
    while not self.EndOfInput():
        self._CheckMapShorthand()
        here = self.GetPosition()
        name = self.Token(_RESERVED_OPERATOR_CHARS, space=False)
        if name:
            is_function = self.IsCharacter('(', peek=True, eoi_ok=True)
            if not key and (not is_function) and (name in self._defaults.aliases):
                k, attribute = self._defaults.aliases[name]
                key.extend(k)
            else:
                key.append(name)
        elif not self.IsCharacter('[', peek=True):
            if not key and self.IsCharacter('.') and (not self.IsCharacter('.', peek=True, eoi_ok=True)) and (self.EndOfInput() or self.IsCharacter(_RESERVED_OPERATOR_CHARS, peek=True, eoi_ok=True)):
                break
            raise resource_exceptions.ExpressionSyntaxError('Non-empty key name expected [{0}].'.format(self.Annotate(here)))
        if self.EndOfInput():
            break
        if self.IsCharacter(']'):
            raise resource_exceptions.ExpressionSyntaxError('Unmatched ] in key [{0}].'.format(self.Annotate(here)))
        while self.IsCharacter('[', eoi_ok=True):
            index = self.Token(']', convert=True)
            self.IsCharacter(']')
            key.append(index)
        if not self.IsCharacter('.', eoi_ok=True):
            break
        if self.EndOfInput():
            raise resource_exceptions.ExpressionSyntaxError('Non-empty key name expected [{0}].'.format(self.Annotate()))
    return (key, attribute)