import functools
from typing import Any, Callable, cast, Optional, Type, TYPE_CHECKING, TypeVar, Union
class GitlabError(Exception):

    def __init__(self, error_message: Union[str, bytes]='', response_code: Optional[int]=None, response_body: Optional[bytes]=None) -> None:
        Exception.__init__(self, error_message)
        self.response_code = response_code
        self.response_body = response_body
        try:
            if TYPE_CHECKING:
                assert isinstance(error_message, bytes)
            self.error_message = error_message.decode()
        except Exception:
            if TYPE_CHECKING:
                assert isinstance(error_message, str)
            self.error_message = error_message

    def __str__(self) -> str:
        if self.response_code is not None:
            return f'{self.response_code}: {self.error_message}'
        return f'{self.error_message}'