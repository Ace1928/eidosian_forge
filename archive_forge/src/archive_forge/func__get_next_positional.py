from __future__ import absolute_import, division, print_function, unicode_literals
import os, sys, argparse, contextlib
from . import completers, my_shlex as shlex
from .compat import USING_PYTHON2, str, sys_encoding, ensure_str, ensure_bytes
from .completers import FilesCompleter, SuppressCompleter
from .my_argparse import IntrospectiveArgumentParser, action_is_satisfied, action_is_open, action_is_greedy
from .shellintegration import shellcode # noqa
def _get_next_positional(self):
    """
        Get the next positional action if it exists.
        """
    active_parser = self.active_parsers[-1]
    last_positional = self.visited_positionals[-1]
    all_positionals = active_parser._get_positional_actions()
    if not all_positionals:
        return None
    if active_parser == last_positional:
        return all_positionals[0]
    i = 0
    for i in range(len(all_positionals)):
        if all_positionals[i] == last_positional:
            break
    if i + 1 < len(all_positionals):
        return all_positionals[i + 1]
    return None