from enum import Enum
from typing import Union, Callable, Optional, cast
class ProviderError(LibcloudError):
    """
    Exception used when provider gives back
    error response (HTTP 4xx, 5xx) for a request.

    Specific sub types can be derived for errors like
    HTTP 401 : InvalidCredsError
    HTTP 404 : NodeNotFoundError, ContainerDoesNotExistError
    """

    def __init__(self, value, http_code, driver=None):
        super().__init__(value=value, driver=driver)
        self.http_code = http_code

    def __str__(self):
        return self.__repr__()

    def __repr__(self):
        return repr(self.value)