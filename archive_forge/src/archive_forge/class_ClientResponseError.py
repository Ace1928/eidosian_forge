import asyncio
import warnings
from typing import TYPE_CHECKING, Any, Optional, Tuple, Union
from .http_parser import RawResponseMessage
from .typedefs import LooseHeaders
class ClientResponseError(ClientError):
    """Base class for exceptions that occur after getting a response.

    request_info: An instance of RequestInfo.
    history: A sequence of responses, if redirects occurred.
    status: HTTP status code.
    message: Error message.
    headers: Response headers.
    """

    def __init__(self, request_info: RequestInfo, history: Tuple[ClientResponse, ...], *, code: Optional[int]=None, status: Optional[int]=None, message: str='', headers: Optional[LooseHeaders]=None) -> None:
        self.request_info = request_info
        if code is not None:
            if status is not None:
                raise ValueError('Both code and status arguments are provided; code is deprecated, use status instead')
            warnings.warn('code argument is deprecated, use status instead', DeprecationWarning, stacklevel=2)
        if status is not None:
            self.status = status
        elif code is not None:
            self.status = code
        else:
            self.status = 0
        self.message = message
        self.headers = headers
        self.history = history
        self.args = (request_info, history)

    def __str__(self) -> str:
        return '{}, message={!r}, url={!r}'.format(self.status, self.message, self.request_info.real_url)

    def __repr__(self) -> str:
        args = f'{self.request_info!r}, {self.history!r}'
        if self.status != 0:
            args += f', status={self.status!r}'
        if self.message != '':
            args += f', message={self.message!r}'
        if self.headers is not None:
            args += f', headers={self.headers!r}'
        return f'{type(self).__name__}({args})'

    @property
    def code(self) -> int:
        warnings.warn('code property is deprecated, use status instead', DeprecationWarning, stacklevel=2)
        return self.status

    @code.setter
    def code(self, value: int) -> None:
        warnings.warn('code property is deprecated, use status instead', DeprecationWarning, stacklevel=2)
        self.status = value