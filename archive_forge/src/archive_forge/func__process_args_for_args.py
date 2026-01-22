import typing as t
from collections import deque
from gettext import gettext as _
from gettext import ngettext
from .exceptions import BadArgumentUsage
from .exceptions import BadOptionUsage
from .exceptions import NoSuchOption
from .exceptions import UsageError
def _process_args_for_args(self, state: ParsingState) -> None:
    pargs, args = _unpack_args(state.largs + state.rargs, [x.nargs for x in self._args])
    for idx, arg in enumerate(self._args):
        arg.process(pargs[idx], state)
    state.largs = args
    state.rargs = []