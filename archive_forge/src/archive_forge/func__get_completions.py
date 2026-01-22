from __future__ import absolute_import, division, print_function, unicode_literals
import os, sys, argparse, contextlib
from . import completers, my_shlex as shlex
from .compat import USING_PYTHON2, str, sys_encoding, ensure_str, ensure_bytes
from .completers import FilesCompleter, SuppressCompleter
from .my_argparse import IntrospectiveArgumentParser, action_is_satisfied, action_is_open, action_is_greedy
from .shellintegration import shellcode # noqa
def _get_completions(self, comp_words, cword_prefix, cword_prequote, last_wordbreak_pos):
    active_parsers = self._patch_argument_parser()
    parsed_args = argparse.Namespace()
    self.completing = True
    if USING_PYTHON2:
        comp_words = [ensure_bytes(word) for word in comp_words]
    try:
        debug('invoking parser with', comp_words[1:])
        with mute_stderr():
            a = self._parser.parse_known_args(comp_words[1:], namespace=parsed_args)
        debug('parsed args:', a)
    except BaseException as e:
        debug('\nexception', type(e), str(e), 'while parsing args')
    self.completing = False
    completions = self.collect_completions(active_parsers, parsed_args, cword_prefix, debug)
    completions = self.filter_completions(completions)
    completions = self.quote_completions(completions, cword_prequote, last_wordbreak_pos)
    return completions