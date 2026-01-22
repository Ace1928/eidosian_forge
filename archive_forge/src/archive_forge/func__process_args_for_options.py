import typing as t
from collections import deque
from gettext import gettext as _
from gettext import ngettext
from .exceptions import BadArgumentUsage
from .exceptions import BadOptionUsage
from .exceptions import NoSuchOption
from .exceptions import UsageError
def _process_args_for_options(self, state: ParsingState) -> None:
    while state.rargs:
        arg = state.rargs.pop(0)
        arglen = len(arg)
        if arg == '--':
            return
        elif arg[:1] in self._opt_prefixes and arglen > 1:
            self._process_opts(arg, state)
        elif self.allow_interspersed_args:
            state.largs.append(arg)
        else:
            state.rargs.insert(0, arg)
            return