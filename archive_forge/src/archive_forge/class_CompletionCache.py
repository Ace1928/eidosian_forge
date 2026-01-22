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
class CompletionCache(object):
    """A per-arg cache of completions for the command line under construction.

  Since we have no details on the compeleted values this cache is only for the
  current command line. This means that async activities by other commands
  (creating files, instances, resources) may not be seen until the current
  command under construction is executed.

  Attributes:
    args: The list of CacheArg args holding the completion state for each arg.
    completer: The completer object.
    command_count: The completer.cli.command_count value for the current cache.
  """

    def __init__(self, completer):
        self.args = []
        self.completer = completer
        self.command_count = _INVALID_COMMAND_COUNT

    def IsValid(self):
        return self.command_count == self.completer.cli.command_count

    def ArgMatch(self, args, index):
        """Returns True if args[index] matches the cache prefix for index."""
        if not self.args[index].IsValid():
            return True
        return args[index].value.startswith(self.args[index].prefix)

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

    def Update(self, args, completions):
        """Updates completions for the last arg in args."""
        self.command_count = self.completer.cli.command_count
        last_arg_index = len(args) - 1
        for i in range(last_arg_index):
            if i >= len(self.args):
                self.args.append(CacheArg(args[i].value, None))
            elif not self.ArgMatch(args, i):
                self.args[i].Invalidate()
        a = args[last_arg_index].value
        if last_arg_index == len(self.args):
            self.args.append(CacheArg(a, completions))
        if not self.args[last_arg_index].IsValid() or not a.startswith(self.args[last_arg_index].prefix) or a.endswith('/'):
            if a.endswith('/'):
                if not self.args[last_arg_index].dirs:
                    self.args[last_arg_index].dirs[''] = (self.args[last_arg_index].prefix, self.args[last_arg_index].completions)
                self.args[last_arg_index].dirs[a[:-1]] = (a, completions)
        if completions and '/' in completions[0][:-1] and ('/' not in a):
            dirs = {}
            for comp in completions:
                if comp.endswith('/'):
                    comp = comp[:-1]
                    mark = '/'
                else:
                    mark = ''
                parts = _Split(comp)
                if mark:
                    parts[-1] += mark
                for i in range(len(parts)):
                    d = '/'.join(parts[:i])
                    if d not in dirs:
                        dirs[d] = []
                    comp = '/'.join(parts[:i + 1])
                    if comp.endswith(':/'):
                        comp += '/'
                    if comp not in dirs[d]:
                        dirs[d].append(comp)
            for d, c in six.iteritems(dirs):
                marked = d
                if marked.endswith(':/'):
                    marked += '/'
                self.args[last_arg_index].dirs[d] = (marked, c)
        else:
            self.args[last_arg_index].prefix = a
            self.args[last_arg_index].completions = completions
        for i in range(last_arg_index + 1, len(self.args)):
            self.args[i].Invalidate()