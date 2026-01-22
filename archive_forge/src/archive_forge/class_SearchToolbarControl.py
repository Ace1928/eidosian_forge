from __future__ import unicode_literals
from ..enums import IncrementalSearchDirection
from .processors import BeforeInput
from .lexers import SimpleLexer
from .dimension import LayoutDimension
from .controls import BufferControl, TokenListControl, UIControl, UIContent
from .containers import Window, ConditionalContainer
from .screen import Char
from .utils import token_list_len
from prompt_toolkit.enums import SEARCH_BUFFER, SYSTEM_BUFFER
from prompt_toolkit.filters import HasFocus, HasArg, HasCompletions, HasValidationError, HasSearch, Always, IsDone
from prompt_toolkit.token import Token
class SearchToolbarControl(BufferControl):
    """
    :param vi_mode: Display '/' and '?' instead of I-search.
    """

    def __init__(self, vi_mode=False):
        token = Token.Toolbar.Search

        def get_before_input(cli):
            if not cli.is_searching:
                text = ''
            elif cli.search_state.direction == IncrementalSearchDirection.BACKWARD:
                text = '?' if vi_mode else 'I-search backward: '
            else:
                text = '/' if vi_mode else 'I-search: '
            return [(token, text)]
        super(SearchToolbarControl, self).__init__(buffer_name=SEARCH_BUFFER, input_processors=[BeforeInput(get_before_input)], default_char=Char(token=token), lexer=SimpleLexer(token=token.Text))