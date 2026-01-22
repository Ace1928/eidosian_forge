import time
from typing import Sequence
import curtsies.events
class ReloadEvent(curtsies.events.Event):
    """Request to rerun REPL session ASAP because imported modules changed"""

    def __init__(self, files_modified: Sequence[str]=('?',)) -> None:
        self.files_modified = files_modified

    def __repr__(self) -> str:
        return '<ReloadEvent from {}>'.format(' & '.join(self.files_modified))