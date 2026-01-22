from __future__ import annotations
from typing import TYPE_CHECKING
from ._ki import LOCALS_KEY_KI_PROTECTION_ENABLED
from ._run import GLOBAL_RUN_CONTEXT
Stop instrumenting the current run loop with the given instrument.

    Args:
      instrument (trio.abc.Instrument): The instrument to de-activate.

    Raises:
      KeyError: if the instrument is not currently active. This could
          occur either because you never added it, or because you added it
          and then it raised an unhandled exception and was automatically
          deactivated.

    