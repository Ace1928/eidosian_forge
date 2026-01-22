from __future__ import absolute_import, division, print_function, unicode_literals
import os, sys, argparse, contextlib
from . import completers, my_shlex as shlex
from .compat import USING_PYTHON2, str, sys_encoding, ensure_str, ensure_bytes
from .completers import FilesCompleter, SuppressCompleter
from .my_argparse import IntrospectiveArgumentParser, action_is_satisfied, action_is_open, action_is_greedy
from .shellintegration import shellcode # noqa
class ExclusiveCompletionFinder(CompletionFinder):

    @staticmethod
    def _action_allowed(action, parser):
        if not CompletionFinder._action_allowed(action, parser):
            return False
        append_classes = (argparse._AppendAction, argparse._AppendConstAction)
        if action._orig_class in append_classes:
            return True
        if action not in parser._seen_non_default_actions:
            return True
        return False