from __future__ import unicode_literals
from .base import Filter
from prompt_toolkit.enums import EditingMode
from prompt_toolkit.key_binding.vi_state import InputMode as ViInputMode
from prompt_toolkit.cache import memoized
@memoized()
class EmacsSelectionMode(Filter):

    def __call__(self, cli):
        return cli.editing_mode == EditingMode.EMACS and cli.current_buffer.selection_state

    def __repr__(self):
        return 'EmacsSelectionMode()'