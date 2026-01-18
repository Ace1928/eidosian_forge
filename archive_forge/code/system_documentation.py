from __future__ import unicode_literals
from prompt_toolkit.contrib.regular_languages.completion import GrammarCompleter
from prompt_toolkit.contrib.regular_languages.compiler import compile
from .filesystem import PathCompleter, ExecutableCompleter

    Completer for system commands.
    