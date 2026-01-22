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
def Lookup(self, args):
    """Returns the cached completions for the last arg in args or None."""
    if not args or not self.IsValid():
        return None
    if len(args) > len(self.args):
        return None
    last_arg_index = len(args) - 1
    for i in range(last_arg_index):
        if not self.ArgMatch(args, i):
            return None
    if not self.args[last_arg_index].IsValid():
        return None
    a = args[last_arg_index].value
    if a.endswith('/'):
        parent = a[:-1]
        self.completer.debug.dir.text(parent)
        prefix, completions = self.args[last_arg_index].dirs.get(parent, (None, None))
        if not completions:
            return None
        self.args[last_arg_index].prefix = prefix
        self.args[last_arg_index].completions = completions
    elif a in self.args[last_arg_index].dirs:
        self.completer.debug.dir.text(_Dirname(a))
        prefix, completions = self.args[last_arg_index].dirs.get(_Dirname(a), (None, None))
        if completions:
            self.args[last_arg_index].prefix = prefix
            self.args[last_arg_index].completions = completions
    if not self.ArgMatch(args, last_arg_index):
        return None
    return [c for c in self.args[last_arg_index].completions if c.startswith(a)]