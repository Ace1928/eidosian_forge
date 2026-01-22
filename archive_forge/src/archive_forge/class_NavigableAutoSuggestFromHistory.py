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
class NavigableAutoSuggestFromHistory(AutoSuggestFromHistory):
    """
    A subclass of AutoSuggestFromHistory that allow navigation to next/previous
    suggestion from history. To do so it remembers the current position, but it
    state need to carefully be cleared on the right events.
    """

    def __init__(self):
        self.skip_lines = 0
        self._connected_apps = []

    def reset_history_position(self, _: Buffer):
        self.skip_lines = 0

    def disconnect(self):
        for pt_app in self._connected_apps:
            text_insert_event = pt_app.default_buffer.on_text_insert
            text_insert_event.remove_handler(self.reset_history_position)

    def connect(self, pt_app: PromptSession):
        self._connected_apps.append(pt_app)
        pt_app.default_buffer.on_text_insert.add_handler(self.reset_history_position)
        pt_app.default_buffer.on_cursor_position_changed.add_handler(self._dismiss)

    def get_suggestion(self, buffer: Buffer, document: Document) -> Optional[Suggestion]:
        text = _get_query(document)
        if text.strip():
            for suggestion, _ in self._find_next_match(text, self.skip_lines, buffer.history):
                return Suggestion(suggestion)
        return None

    def _dismiss(self, buffer, *args, **kwargs):
        buffer.suggestion = None

    def _find_match(self, text: str, skip_lines: float, history: History, previous: bool) -> Generator[Tuple[str, float], None, None]:
        """
        text : str
            Text content to find a match for, the user cursor is most of the
            time at the end of this text.
        skip_lines : float
            number of items to skip in the search, this is used to indicate how
            far in the list the user has navigated by pressing up or down.
            The float type is used as the base value is +inf
        history : History
            prompt_toolkit History instance to fetch previous entries from.
        previous : bool
            Direction of the search, whether we are looking previous match
            (True), or next match (False).

        Yields
        ------
        Tuple with:
        str:
            current suggestion.
        float:
            will actually yield only ints, which is passed back via skip_lines,
            which may be a +inf (float)


        """
        line_number = -1
        for string in reversed(list(history.get_strings())):
            for line in reversed(string.splitlines()):
                line_number += 1
                if not previous and line_number < skip_lines:
                    continue
                if line.startswith(text) and len(line) > len(text):
                    yield (line[len(text):], line_number)
                if previous and line_number >= skip_lines:
                    return

    def _find_next_match(self, text: str, skip_lines: float, history: History) -> Generator[Tuple[str, float], None, None]:
        return self._find_match(text, skip_lines, history, previous=False)

    def _find_previous_match(self, text: str, skip_lines: float, history: History):
        return reversed(list(self._find_match(text, skip_lines, history, previous=True)))

    def up(self, query: str, other_than: str, history: History) -> None:
        for suggestion, line_number in self._find_next_match(query, self.skip_lines, history):
            if query + suggestion != other_than:
                self.skip_lines = line_number
                break
        else:
            self.skip_lines = 0

    def down(self, query: str, other_than: str, history: History) -> None:
        for suggestion, line_number in self._find_previous_match(query, self.skip_lines, history):
            if query + suggestion != other_than:
                self.skip_lines = line_number
                break
        else:
            for suggestion, line_number in self._find_previous_match(query, float('Inf'), history):
                if query + suggestion != other_than:
                    self.skip_lines = line_number
                    break