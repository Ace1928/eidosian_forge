import logging
import types
from typing import Any, Callable, Dict, Sequence, TypeVar
from .._abc import Instrument
@_public
def add_instrument(self, instrument: Instrument) -> None:
    """Start instrumenting the current run loop with the given instrument.

        Args:
          instrument (trio.abc.Instrument): The instrument to activate.

        If ``instrument`` is already active, does nothing.

        """
    if instrument in self['_all']:
        return
    self['_all'][instrument] = None
    try:
        for name in dir(instrument):
            if name.startswith('_'):
                continue
            try:
                prototype = getattr(Instrument, name)
            except AttributeError:
                continue
            impl = getattr(instrument, name)
            if isinstance(impl, types.MethodType) and impl.__func__ is prototype:
                continue
            self.setdefault(name, {})[instrument] = None
    except:
        self.remove_instrument(instrument)
        raise