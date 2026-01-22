import ast
import re
import signal
import sys
from typing import Callable, Dict, Union
from prompt_toolkit.application.current import get_app
from prompt_toolkit.enums import DEFAULT_BUFFER, SEARCH_BUFFER
from prompt_toolkit.key_binding import KeyPressEvent
from prompt_toolkit.filters import Condition, Filter, emacs_insert_mode, has_completions
from prompt_toolkit.filters import has_focus as has_focus_impl
from prompt_toolkit.filters import (
from prompt_toolkit.layout.layout import FocusableElement
from IPython.core.getipython import get_ipython
from IPython.core.guarded_eval import _find_dunder, BINARY_OP_DUNDERS, UNARY_OP_DUNDERS
from IPython.terminal.shortcuts import auto_suggest
from IPython.utils.decorators import undoc
def _preceding_text():
    app = get_app()
    before_cursor = app.current_buffer.document.current_line_before_cursor
    return bool(m.match(before_cursor))