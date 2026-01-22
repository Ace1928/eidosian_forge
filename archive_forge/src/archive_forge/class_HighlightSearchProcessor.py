from __future__ import unicode_literals
from abc import ABCMeta, abstractmethod
from six import with_metaclass
from six.moves import range
from prompt_toolkit.cache import SimpleCache
from prompt_toolkit.document import Document
from prompt_toolkit.enums import SEARCH_BUFFER
from prompt_toolkit.filters import to_cli_filter, ViInsertMultipleMode
from prompt_toolkit.layout.utils import token_list_to_text
from prompt_toolkit.reactive import Integer
from prompt_toolkit.token import Token
from .utils import token_list_len, explode_tokens
import re
class HighlightSearchProcessor(Processor):
    """
    Processor that highlights search matches in the document.
    Note that this doesn't support multiline search matches yet.

    :param preview_search: A Filter; when active it indicates that we take
        the search text in real time while the user is typing, instead of the
        last active search state.
    """

    def __init__(self, preview_search=False, search_buffer_name=SEARCH_BUFFER, get_search_state=None):
        self.preview_search = to_cli_filter(preview_search)
        self.search_buffer_name = search_buffer_name
        self.get_search_state = get_search_state or (lambda cli: cli.search_state)

    def _get_search_text(self, cli):
        """
        The text we are searching for.
        """
        if self.preview_search(cli) and cli.buffers[self.search_buffer_name].text:
            return cli.buffers[self.search_buffer_name].text
        else:
            return self.get_search_state(cli).text

    def apply_transformation(self, cli, document, lineno, source_to_display, tokens):
        search_text = self._get_search_text(cli)
        searchmatch_current_token = (':',) + Token.SearchMatch.Current
        searchmatch_token = (':',) + Token.SearchMatch
        if search_text and (not cli.is_returning):
            line_text = token_list_to_text(tokens)
            tokens = explode_tokens(tokens)
            flags = re.IGNORECASE if cli.is_ignoring_case else 0
            if document.cursor_position_row == lineno:
                cursor_column = source_to_display(document.cursor_position_col)
            else:
                cursor_column = None
            for match in re.finditer(re.escape(search_text), line_text, flags=flags):
                if cursor_column is not None:
                    on_cursor = match.start() <= cursor_column < match.end()
                else:
                    on_cursor = False
                for i in range(match.start(), match.end()):
                    old_token, text = tokens[i]
                    if on_cursor:
                        tokens[i] = (old_token + searchmatch_current_token, tokens[i][1])
                    else:
                        tokens[i] = (old_token + searchmatch_token, tokens[i][1])
        return Transformation(tokens)