from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import enum
from googlecloudsdk.calliope import cli_tree
from googlecloudsdk.command_lib.interactive import lexer
import six
def ParseFlag(self, token, name):
    """Parses the flag token and appends it to the arg list."""
    name_start = token.start
    name_end = token.end
    value = None
    value_start = None
    value_end = None
    if '=' in name:
        name, value = name.split('=', 1)
        name_end = name_start + len(name)
        value_start = name_end + 1
        value_end = value_start + len(value)
    flag = self.cmd[LOOKUP_FLAGS].get(name)
    if not flag or (not self.hidden and flag[LOOKUP_IS_HIDDEN]):
        self.args.append(ArgToken(name, ArgTokenType.UNKNOWN, self.cmd, token.start, token.end))
        return
    if flag[LOOKUP_TYPE] != 'bool' and value is None and self.tokens:
        token = self.tokens.pop(0)
        value = token.UnquotedValue()
        value_start = token.start
        value_end = token.end
    self.args.append(ArgToken(name, ArgTokenType.FLAG, flag, name_start, name_end))
    if value is not None:
        self.args.append(ArgToken(value, ArgTokenType.FLAG_ARG, None, value_start, value_end))