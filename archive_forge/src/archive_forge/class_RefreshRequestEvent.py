import time
from typing import Sequence
import curtsies.events
class RefreshRequestEvent(curtsies.events.Event):
    """Request to refresh REPL display ASAP"""

    def __repr__(self) -> str:
        return '<RefreshRequestEvent for now>'