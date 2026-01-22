from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import io
import os
import sys
import threading
import time
from googlecloudsdk.calliope import parser_completer
from googlecloudsdk.command_lib.interactive import parser
from googlecloudsdk.command_lib.meta import generate_cli_trees
from googlecloudsdk.core import module_util
from googlecloudsdk.core.console import console_attr
from prompt_toolkit import completion
import six
def CommandCompleter(self, args):
    """Returns the command/group completion choices for args or None.

    Args:
      args: The CLI tree parsed command args.

    Returns:
      (choices, offset):
        choices - The list of completion strings or None.
        offset - The completion prefix offset.
    """
    arg = args[-1]
    if arg.value.startswith('-'):
        return (None, 0)
    elif self.IsPrefixArg(args):
        node = self.parser.root
        prefix = arg.value
    elif arg.token_type in (parser.ArgTokenType.COMMAND, parser.ArgTokenType.GROUP) and (not self.empty):
        node = args[-2].tree if len(args) > 1 else self.parser.root
        prefix = arg.value
    elif arg.token_type == parser.ArgTokenType.GROUP:
        if not self.empty:
            return ([], 0)
        node = arg.tree
        prefix = ''
    elif arg.token_type == parser.ArgTokenType.UNKNOWN:
        prefix = arg.value
        if self.manpage_generator and (not prefix) and (len(args) == 2) and args[0].value:
            node = generate_cli_trees.LoadOrGenerate(args[0].value)
            if not node:
                return (None, 0)
            self.parser.root[parser.LOOKUP_COMMANDS][args[0].value] = node
        elif len(args) > 1 and args[-2].token_type == parser.ArgTokenType.GROUP:
            node = args[-2].tree
        else:
            return (None, 0)
    else:
        return (None, 0)
    choices = [k for k, v in six.iteritems(node[parser.LOOKUP_COMMANDS]) if k.startswith(prefix) and (not self.IsSuppressed(v))]
    if choices:
        return (choices, -len(prefix))
    return (None, 0)