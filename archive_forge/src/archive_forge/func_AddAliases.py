from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.core.cache import cache_update_ops
def AddAliases(self, aliases):
    """Adds aliases to the display info, newer values takes precedence.

    Args:
      aliases: {str: parsed_key} The resource name alias dict that maps short
        names to parsed keys. The parsed key for 'abc.xyz' is ['abc', 'xyz'].
        See the resource_lex.Lexer.Key() docstring for parsed key details.
    """
    if aliases:
        self._aliases.update(aliases)