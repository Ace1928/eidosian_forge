from __future__ import unicode_literals
from six import text_type
from prompt_toolkit.enums import IncrementalSearchDirection, SEARCH_BUFFER
from prompt_toolkit.token import Token
from .utils import token_list_len
from .processors import Processor, Transformation
class DefaultPrompt(Processor):
    """
    Default prompt. This one shows the 'arg' and reverse search like
    Bash/readline normally do.

    There are two ways to instantiate a ``DefaultPrompt``. For a prompt
    with a static message, do for instance::

        prompt = DefaultPrompt.from_message('prompt> ')

    For a dynamic prompt, generated from a token list function::

        def get_tokens(cli):
            return [(Token.A, 'text'), (Token.B, 'text2')]

        prompt = DefaultPrompt(get_tokens)
    """

    def __init__(self, get_tokens):
        assert callable(get_tokens)
        self.get_tokens = get_tokens

    @classmethod
    def from_message(cls, message='> '):
        """
        Create a default prompt with a static message text.
        """
        assert isinstance(message, text_type)

        def get_message_tokens(cli):
            return [(Token.Prompt, message)]
        return cls(get_message_tokens)

    def apply_transformation(self, cli, document, lineno, source_to_display, tokens):
        if cli.is_searching:
            before = _get_isearch_tokens(cli)
        elif cli.input_processor.arg is not None:
            before = _get_arg_tokens(cli)
        else:
            before = self.get_tokens(cli)
        shift_position = token_list_len(before)
        if lineno != 0:
            before = [(Token.Prompt, ' ' * shift_position)]
        return Transformation(tokens=before + tokens, source_to_display=lambda i: i + shift_position, display_to_source=lambda i: i - shift_position)

    def has_focus(self, cli):
        return cli.is_searching