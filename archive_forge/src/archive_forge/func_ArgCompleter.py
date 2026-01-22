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
def ArgCompleter(self, args, arg, value):
    """Returns the flag or positional completion choices for arg or [].

    Args:
      args: The CLI tree parsed command args.
      arg: The flag or positional argument.
      value: The (partial) arg value.

    Returns:
      (choices, offset):
        choices - The list of completion strings or None.
        offset - The completion prefix offset.
    """
    choices = arg.get(parser.LOOKUP_CHOICES)
    if choices:
        return ([v for v in choices if v.startswith(value)], -len(value))
    if not value and (not self.event.completion_requested):
        return ([], 0)
    module_path = arg.get(parser.LOOKUP_COMPLETER)
    if not module_path:
        return ([], 0)
    cache = self.module_cache.get(module_path)
    if not cache:
        cache = ModuleCache(module_util.ImportModule(module_path))
        self.module_cache[module_path] = cache
    prefix = value
    if not isinstance(cache.completer_class, type):
        cache.choices = cache.completer_class(prefix=prefix)
    elif cache.stale < time.time():
        old_dict = self.parsed_args.__dict__
        self.parsed_args.__dict__ = {}
        self.parsed_args.__dict__.update(old_dict)
        self.parsed_args.__dict__.update(_NameSpaceDict(args))
        completer = parser_completer.ArgumentCompleter(cache.completer_class, parsed_args=self.parsed_args)
        with Spinner(self.SetSpinner):
            cache.choices = completer(prefix='')
        self.parsed_args.__dict__ = old_dict
        cache.stale = time.time() + cache.timeout
    if arg.get(parser.LOOKUP_TYPE) == 'list':
        parts = value.split(',')
        prefix = parts[-1]
    if not cache.choices:
        return ([], 0)
    return ([v for v in cache.choices if v.startswith(prefix)], -len(prefix))