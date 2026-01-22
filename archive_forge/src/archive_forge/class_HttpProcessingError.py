from textwrap import indent
from typing import Optional, Union
from .typedefs import _CIMultiDict
class HttpProcessingError(Exception):
    """HTTP error.

    Shortcut for raising HTTP errors with custom code, message and headers.

    code: HTTP Error code.
    message: (optional) Error message.
    headers: (optional) Headers to be sent in response, a list of pairs
    """
    code = 0
    message = ''
    headers = None

    def __init__(self, *, code: Optional[int]=None, message: str='', headers: Optional[_CIMultiDict]=None) -> None:
        if code is not None:
            self.code = code
        self.headers = headers
        self.message = message

    def __str__(self) -> str:
        msg = indent(self.message, '  ')
        return f'{self.code}, message:\n{msg}'

    def __repr__(self) -> str:
        return f'<{self.__class__.__name__}: {self.code}, message={self.message!r}>'