from typing import Optional
class CommError(Error):
    """Error communicating with W&B servers."""

    def __init__(self, msg, exc=None) -> None:
        self.exc = exc
        self.message = msg
        super().__init__(self.message)