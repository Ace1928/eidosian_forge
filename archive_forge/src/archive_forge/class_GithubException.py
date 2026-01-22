import json
from typing import Any, Dict, List, Optional, Tuple, Type, Union
class GithubException(Exception):
    """
    Error handling in PyGithub is done with exceptions. This class is the base of all exceptions raised by PyGithub (but :class:`github.GithubException.BadAttributeException`).

    Some other types of exceptions might be raised by underlying libraries, for example for network-related issues.
    """

    def __init__(self, status: int, data: Any=None, headers: Optional[Dict[str, str]]=None, message: Optional[str]=None):
        super().__init__()
        self.__status = status
        self.__data = data
        self.__headers = headers
        self.__message = message
        self.args = (status, data, headers, message)

    @property
    def message(self) -> Optional[str]:
        return self.__message

    @property
    def status(self) -> int:
        """
        The status returned by the Github API
        """
        return self.__status

    @property
    def data(self) -> Any:
        """
        The (decoded) data returned by the Github API
        """
        return self.__data

    @property
    def headers(self) -> Optional[Dict[str, str]]:
        """
        The headers returned by the Github API
        """
        return self.__headers

    def __repr__(self) -> str:
        return f'{self.__class__.__name__}({self.__str__()})'

    def __str__(self) -> str:
        if self.__message:
            msg = f'{self.__message}: {self.status}'
        else:
            msg = f'{self.status}'
        if self.data is not None:
            msg += ' ' + json.dumps(self.data)
        return msg