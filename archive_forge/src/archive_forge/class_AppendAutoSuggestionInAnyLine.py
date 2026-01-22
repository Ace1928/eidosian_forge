import re
import tokenize
from io import StringIO
from typing import Callable, List, Optional, Union, Generator, Tuple
import warnings
from prompt_toolkit.buffer import Buffer
from prompt_toolkit.key_binding import KeyPressEvent
from prompt_toolkit.key_binding.bindings import named_commands as nc
from prompt_toolkit.auto_suggest import AutoSuggestFromHistory, Suggestion
from prompt_toolkit.document import Document
from prompt_toolkit.history import History
from prompt_toolkit.shortcuts import PromptSession
from prompt_toolkit.layout.processors import (
from IPython.core.getipython import get_ipython
from IPython.utils.tokenutil import generate_tokens
from .filters import pass_through
class AppendAutoSuggestionInAnyLine(Processor):
    """
    Append the auto suggestion to lines other than the last (appending to the
    last line is natively supported by the prompt toolkit).
    """

    def __init__(self, style: str='class:auto-suggestion') -> None:
        self.style = style

    def apply_transformation(self, ti: TransformationInput) -> Transformation:
        is_last_line = ti.lineno == ti.document.line_count - 1
        is_active_line = ti.lineno == ti.document.cursor_position_row
        if not is_last_line and is_active_line:
            buffer = ti.buffer_control.buffer
            if buffer.suggestion and ti.document.is_cursor_at_the_end_of_line:
                suggestion = buffer.suggestion.text
            else:
                suggestion = ''
            return Transformation(fragments=ti.fragments + [(self.style, suggestion)])
        else:
            return Transformation(fragments=ti.fragments)