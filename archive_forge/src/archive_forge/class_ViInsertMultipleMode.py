from __future__ import unicode_literals
from .base import Filter
from prompt_toolkit.enums import EditingMode
from prompt_toolkit.key_binding.vi_state import InputMode as ViInputMode
from prompt_toolkit.cache import memoized
@memoized()
class ViInsertMultipleMode(Filter):

    def __call__(self, cli):
        if cli.editing_mode != EditingMode.VI or cli.vi_state.operator_func or cli.vi_state.waiting_for_digraph or cli.current_buffer.selection_state or cli.current_buffer.read_only():
            return False
        return cli.vi_state.input_mode == ViInputMode.INSERT_MULTIPLE

    def __repr__(self):
        return 'ViInsertMultipleMode()'