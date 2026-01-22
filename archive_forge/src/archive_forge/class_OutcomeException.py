import sys
from typing import Any
from typing import Callable
from typing import cast
from typing import NoReturn
from typing import Optional
from typing import Protocol
from typing import Type
from typing import TypeVar
class OutcomeException(BaseException):
    """OutcomeException and its subclass instances indicate and contain info
    about test and collection outcomes."""

    def __init__(self, msg: Optional[str]=None, pytrace: bool=True) -> None:
        if msg is not None and (not isinstance(msg, str)):
            error_msg = "{} expected string as 'msg' parameter, got '{}' instead.\nPerhaps you meant to use a mark?"
            raise TypeError(error_msg.format(type(self).__name__, type(msg).__name__))
        super().__init__(msg)
        self.msg = msg
        self.pytrace = pytrace

    def __repr__(self) -> str:
        if self.msg is not None:
            return self.msg
        return f'<{self.__class__.__name__} instance>'
    __str__ = __repr__