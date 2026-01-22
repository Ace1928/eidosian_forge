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
class TabsProcessor(Processor):
    """
    Render tabs as spaces (instead of ^I) or make them visible (for instance,
    by replacing them with dots.)

    :param tabstop: (Integer) Horizontal space taken by a tab.
    :param get_char1: Callable that takes a `CommandLineInterface` and return a
        character (text of length one). This one is used for the first space
        taken by the tab.
    :param get_char2: Like `get_char1`, but for the rest of the space.
    """

    def __init__(self, tabstop=4, get_char1=None, get_char2=None, token=Token.Tab):
        assert isinstance(tabstop, Integer)
        assert get_char1 is None or callable(get_char1)
        assert get_char2 is None or callable(get_char2)
        self.get_char1 = get_char1 or get_char2 or (lambda cli: '|')
        self.get_char2 = get_char2 or get_char1 or (lambda cli: '┈')
        self.tabstop = tabstop
        self.token = token

    def apply_transformation(self, cli, document, lineno, source_to_display, tokens):
        tabstop = int(self.tabstop)
        token = self.token
        separator1 = self.get_char1(cli)
        separator2 = self.get_char2(cli)
        tokens = explode_tokens(tokens)
        position_mappings = {}
        result_tokens = []
        pos = 0
        for i, token_and_text in enumerate(tokens):
            position_mappings[i] = pos
            if token_and_text[1] == '\t':
                count = tabstop - pos % tabstop
                if count == 0:
                    count = tabstop
                result_tokens.append((token, separator1))
                result_tokens.append((token, separator2 * (count - 1)))
                pos += count
            else:
                result_tokens.append(token_and_text)
                pos += 1
        position_mappings[len(tokens)] = pos

        def source_to_display(from_position):
            """ Maps original cursor position to the new one. """
            return position_mappings[from_position]

        def display_to_source(display_pos):
            """ Maps display cursor position to the original one. """
            position_mappings_reversed = dict(((v, k) for k, v in position_mappings.items()))
            while display_pos >= 0:
                try:
                    return position_mappings_reversed[display_pos]
                except KeyError:
                    display_pos -= 1
            return 0
        return Transformation(result_tokens, source_to_display=source_to_display, display_to_source=display_to_source)