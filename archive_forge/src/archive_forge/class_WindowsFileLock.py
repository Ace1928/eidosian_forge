from typing import Optional, Type, IO
from ._typing_compat import Literal
from types import TracebackType
class WindowsFileLock(BaseLock):
    """Simple file locking for Windows using msvcrt"""

    def __init__(self, filename: str) -> None:
        super().__init__()
        self.filename = f'{filename}.lock'
        self.fileobj = -1

    def acquire(self) -> None:
        self.fileobj = os.open(self.filename, os.O_RDWR | os.O_CREAT | os.O_TRUNC)
        msvcrt.locking(self.fileobj, msvcrt.LK_NBLCK, 1)
        self.locked = True

    def release(self) -> None:
        self.locked = False
        msvcrt.locking(self.fileobj, msvcrt.LK_UNLCK, 1)
        os.close(self.fileobj)
        self.fileobj = -1
        try:
            os.remove(self.filename)
        except OSError:
            pass