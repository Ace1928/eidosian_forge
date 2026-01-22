import os
import sys
import inspect
from warnings import warn
from typing import Union as UnionType, Optional
from IPython.core.async_helpers import get_asyncio_loop
from IPython.core.interactiveshell import InteractiveShell, InteractiveShellABC
from IPython.utils.py3compat import input
from IPython.utils.terminal import toggle_set_term_title, set_term_title, restore_term_title
from IPython.utils.process import abbrev_cwd
from traitlets import (
from prompt_toolkit.auto_suggest import AutoSuggestFromHistory
from prompt_toolkit.enums import DEFAULT_BUFFER, EditingMode
from prompt_toolkit.filters import HasFocus, Condition, IsDone
from prompt_toolkit.formatted_text import PygmentsTokens
from prompt_toolkit.history import History
from prompt_toolkit.layout.processors import ConditionalProcessor, HighlightMatchingBracketProcessor
from prompt_toolkit.output import ColorDepth
from prompt_toolkit.patch_stdout import patch_stdout
from prompt_toolkit.shortcuts import PromptSession, CompleteStyle, print_formatted_text
from prompt_toolkit.styles import DynamicStyle, merge_styles
from prompt_toolkit.styles.pygments import style_from_pygments_cls, style_from_pygments_dict
from prompt_toolkit import __version__ as ptk_version
from pygments.styles import get_style_by_name
from pygments.style import Style
from pygments.token import Token
from .debugger import TerminalPdb, Pdb
from .magics import TerminalMagics
from .pt_inputhooks import get_inputhook_name_and_func
from .prompts import Prompts, ClassicPrompts, RichPromptDisplayHook
from .ptutils import IPythonPTCompleter, IPythonPTLexer
from .shortcuts import (
from .shortcuts.filters import KEYBINDING_FILTERS, filter_from_string
from .shortcuts.auto_suggest import (
def _extra_prompt_options(self):
    """
        Return the current layout option for the current Terminal InteractiveShell
        """

    def get_message():
        return PygmentsTokens(self.prompts.in_prompt_tokens())
    if self.editing_mode == 'emacs' and self.prompt_line_number_format == '':
        get_message = get_message()
    options = {'complete_in_thread': False, 'lexer': IPythonPTLexer(), 'reserve_space_for_menu': self.space_for_menu, 'message': get_message, 'prompt_continuation': lambda width, lineno, is_soft_wrap: PygmentsTokens(_backward_compat_continuation_prompt_tokens(self.prompts.continuation_prompt_tokens, width, lineno=lineno)), 'multiline': True, 'complete_style': self.pt_complete_style, 'input_processors': [ConditionalProcessor(processor=HighlightMatchingBracketProcessor(chars='[](){}'), filter=HasFocus(DEFAULT_BUFFER) & ~IsDone() & Condition(lambda: self.highlight_matching_brackets)), ConditionalProcessor(processor=AppendAutoSuggestionInAnyLine(), filter=HasFocus(DEFAULT_BUFFER) & ~IsDone() & Condition(lambda: isinstance(self.auto_suggest, NavigableAutoSuggestFromHistory)))]}
    if not PTK3:
        options['inputhook'] = self.inputhook
    return options