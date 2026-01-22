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
class CompletionsToolbar(ConditionalContainer):

    def __init__(self, extra_filter=Always()):
        super(CompletionsToolbar, self).__init__(content=Window(CompletionsToolbarControl(), height=LayoutDimension.exact(1)), filter=HasCompletions() & ~IsDone() & extra_filter)