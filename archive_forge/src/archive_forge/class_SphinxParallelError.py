from typing import Any, Optional
class SphinxParallelError(SphinxError):
    """Sphinx parallel build error."""
    category = 'Sphinx parallel build error'

    def __init__(self, message: str, traceback: Any) -> None:
        self.message = message
        self.traceback = traceback

    def __str__(self) -> str:
        return self.message