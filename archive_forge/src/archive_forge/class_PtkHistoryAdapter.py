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
class PtkHistoryAdapter(History):
    """
    Prompt toolkit has it's own way of handling history, Where it assumes it can
    Push/pull from history.

    """

    def __init__(self, shell):
        super().__init__()
        self.shell = shell
        self._refresh()

    def append_string(self, string):
        self._loaded = False
        self._refresh()

    def _refresh(self):
        if not self._loaded:
            self._loaded_strings = list(self.load_history_strings())

    def load_history_strings(self):
        last_cell = ''
        res = []
        for __, ___, cell in self.shell.history_manager.get_tail(self.shell.history_load_length, include_latest=True):
            cell = cell.rstrip()
            if cell and cell != last_cell:
                res.append(cell)
                last_cell = cell
        yield from res[::-1]

    def store_string(self, string: str) -> None:
        pass