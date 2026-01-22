from typing import Optional, Type, IO
from ._typing_compat import Literal
from types import TracebackType
class UnixFileLock(BaseLock):
    """Simple file locking for Unix using fcntl"""

    def __init__(self, fileobj, mode: int=0) -> None:
        super().__init__()
        self.fileobj = fileobj
        self.mode = mode | fcntl.LOCK_EX

    def acquire(self) -> None:
        try:
            fcntl.flock(self.fileobj, self.mode)
            self.locked = True
        except OSError as e:
            if e.errno != errno.ENOLCK:
                raise e

    def release(self) -> None:
        self.locked = False
        fcntl.flock(self.fileobj, fcntl.LOCK_UN)