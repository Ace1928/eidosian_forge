import threading
from typing import Any, Dict, Optional, Sequence
from uuid import UUID
from langchain_core.callbacks import base as base_callbacks
from langchain_core.documents import Document
from langchain_core.outputs import LLMResult
def _print_bar(self) -> None:
    """Print the progress bar to the console."""
    progress = self.counter / self.total
    arrow = '-' * int(round(progress * self.ncols) - 1) + '>'
    spaces = ' ' * (self.ncols - len(arrow))
    print(f'\r[{arrow + spaces}] {self.counter}/{self.total}', end='')