from __future__ import unicode_literals
from .base import Filter
from prompt_toolkit.enums import EditingMode
from prompt_toolkit.key_binding.vi_state import InputMode as ViInputMode
from prompt_toolkit.cache import memoized
@memoized()
class ViMode(Filter):

    def __call__(self, cli):
        return cli.editing_mode == EditingMode.VI

    def __repr__(self):
        return 'ViMode()'