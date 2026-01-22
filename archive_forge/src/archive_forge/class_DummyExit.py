from io import StringIO
from typing import Optional, Union
import twisted.trial.unittest
from ...runner import _exit
from .._exit import ExitStatus, exit
class DummyExit:
    """
    Stub for L{sys.exit} that remembers whether it's been called and, if it
    has, what argument it was given.
    """

    def __init__(self) -> None:
        self.exited = False

    def __call__(self, arg: Optional[Union[int, str]]=None) -> None:
        assert not self.exited
        self.arg = arg
        self.exited = True