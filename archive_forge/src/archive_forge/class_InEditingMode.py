from __future__ import unicode_literals
from .base import Filter
from prompt_toolkit.enums import EditingMode
from prompt_toolkit.key_binding.vi_state import InputMode as ViInputMode
from prompt_toolkit.cache import memoized
@memoized()
class InEditingMode(Filter):
    """
    Check whether a given editing mode is active. (Vi or Emacs.)
    """

    def __init__(self, editing_mode):
        self._editing_mode = editing_mode

    @property
    def editing_mode(self):
        """ The given editing mode. (Read-only) """
        return self._editing_mode

    def __call__(self, cli):
        return cli.editing_mode == self.editing_mode

    def __repr__(self):
        return 'InEditingMode(%r)' % (self.editing_mode,)