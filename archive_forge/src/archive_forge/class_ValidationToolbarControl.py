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
class ValidationToolbarControl(TokenListControl):

    def __init__(self, show_position=False):
        token = Token.Toolbar.Validation

        def get_tokens(cli):
            buffer = cli.current_buffer
            if buffer.validation_error:
                row, column = buffer.document.translate_index_to_position(buffer.validation_error.cursor_position)
                if show_position:
                    text = '%s (line=%s column=%s)' % (buffer.validation_error.message, row + 1, column + 1)
                else:
                    text = buffer.validation_error.message
                return [(token, text)]
            else:
                return []
        super(ValidationToolbarControl, self).__init__(get_tokens)