import unicodedata
from wcwidth import wcwidth
from IPython.core.completer import (
from prompt_toolkit.completion import Completer, Completion
from prompt_toolkit.lexers import Lexer
from prompt_toolkit.lexers import PygmentsLexer
from prompt_toolkit.patch_stdout import patch_stdout
import pygments.lexers as pygments_lexers
import os
import sys
import traceback
def _adjust_completion_text_based_on_context(text, body, offset):
    if text.endswith('=') and len(body) > offset and (body[offset] == '='):
        return text[:-1]
    else:
        return text