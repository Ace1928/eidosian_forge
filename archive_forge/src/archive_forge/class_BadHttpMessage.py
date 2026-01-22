from textwrap import indent
from typing import Optional, Union
from .typedefs import _CIMultiDict
class BadHttpMessage(HttpProcessingError):
    code = 400
    message = 'Bad Request'

    def __init__(self, message: str, *, headers: Optional[_CIMultiDict]=None) -> None:
        super().__init__(message=message, headers=headers)
        self.args = (message,)