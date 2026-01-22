from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import argparse
import collections
import os
import re
import sys
import types
import uuid
import argcomplete
from googlecloudsdk.calliope import actions
from googlecloudsdk.calliope import backend
from googlecloudsdk.calliope import base as calliope_base
from googlecloudsdk.calliope import command_loading
from googlecloudsdk.calliope import exceptions
from googlecloudsdk.calliope import parser_errors
from googlecloudsdk.calliope import parser_extensions
from googlecloudsdk.core import argv_utils
from googlecloudsdk.core import config
from googlecloudsdk.core import log
from googlecloudsdk.core import metrics
from googlecloudsdk.core import properties
from googlecloudsdk.core import yaml
from googlecloudsdk.core.configurations import named_configs
from googlecloudsdk.core.console import console_attr
from googlecloudsdk.core.util import encoding
from googlecloudsdk.core.util import files
from googlecloudsdk.core.util import pkg_resources
import six
class _CompletionFinder(argcomplete.CompletionFinder):
    """Calliope overrides for argcomplete.CompletionFinder.

  This makes calliope ArgumentInterceptor and actions objects visible to the
  argcomplete monkeypatcher.
  """

    def _patch_argument_parser(self):
        ai = self._parser
        self._parser = ai.parser
        active_parsers = super(_CompletionFinder, self)._patch_argument_parser()
        if ai:
            self._parser = ai
        return active_parsers

    def _get_completions(self, comp_words, cword_prefix, cword_prequote, last_wordbreak_pos):
        active_parsers = self._patch_argument_parser()
        parsed_args = parser_extensions.Namespace()
        self.completing = True
        try:
            self._parser.parse_known_args(comp_words[1:], namespace=parsed_args)
        except BaseException:
            pass
        self.completing = False
        completions = self.collect_completions(active_parsers, parsed_args, cword_prefix, lambda *_: None)
        completions = self.filter_completions(completions)
        return self.quote_completions(completions, cword_prequote, last_wordbreak_pos)

    def quote_completions(self, completions, cword_prequote, last_wordbreak_pos):
        """Returns the completion (less aggressively) quoted for the shell.

    If the word under the cursor started with a quote (as indicated by a
    nonempty ``cword_prequote``), escapes occurrences of that quote character
    in the completions, and adds the quote to the beginning of each completion.
    Otherwise, escapes *most* characters that bash splits words on
    (``COMP_WORDBREAKS``), and removes portions of completions before the first
    colon if (``COMP_WORDBREAKS``) contains a colon.

    If there is only one completion, and it doesn't end with a
    **continuation character** (``/``, ``:``, or ``=``), adds a space after
    the completion.

    Args:
      completions: The current completion strings.
      cword_prequote: The current quote character in progress, '' if none.
      last_wordbreak_pos: The index of the last wordbreak.

    Returns:
      The completions quoted for the shell.
    """
        no_quote_special = '\\();<>|&$* \t\n`"\''
        double_quote_special = '\\`"$'
        single_quote_special = '\\'
        continuation_special = '=/:'
        no_escaping_shells = ('tcsh', 'fish', 'zsh')
        if not cword_prequote:
            if last_wordbreak_pos:
                completions = [c[last_wordbreak_pos + 1:] for c in completions]
            special_chars = no_quote_special
        elif cword_prequote == '"':
            special_chars = double_quote_special
        else:
            special_chars = single_quote_special
        if encoding.GetEncodedValue(os.environ, '_ARGCOMPLETE_SHELL') in no_escaping_shells:
            special_chars = ''
        elif cword_prequote == "'":
            special_chars = ''
            completions = [c.replace("'", "'\\''") for c in completions]
        for char in special_chars:
            completions = [c.replace(char, '\\' + char) for c in completions]
        if getattr(self, 'append_space', False):
            continuation_chars = continuation_special
            if len(completions) == 1 and completions[0][-1] not in continuation_chars:
                if not cword_prequote:
                    completions[0] += ' '
        return completions