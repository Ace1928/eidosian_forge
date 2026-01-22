import typing
from abc import ABCMeta
from abc import abstractmethod
from enum import Enum
from selenium.common.exceptions import InvalidArgumentException
from selenium.webdriver.common.proxy import Proxy
class ArgOptions(BaseOptions):
    BINARY_LOCATION_ERROR = 'Binary Location Must be a String'

    def __init__(self) -> None:
        super().__init__()
        self._arguments = []
        self._ignore_local_proxy = False

    @property
    def arguments(self):
        """:Returns: A list of arguments needed for the browser."""
        return self._arguments

    def add_argument(self, argument) -> None:
        """Adds an argument to the list.

        :Args:
         - Sets the arguments
        """
        if argument:
            self._arguments.append(argument)
        else:
            raise ValueError('argument can not be null')

    def ignore_local_proxy_environment_variables(self) -> None:
        """By calling this you will ignore HTTP_PROXY and HTTPS_PROXY from
        being picked up and used."""
        self._ignore_local_proxy = True

    def to_capabilities(self):
        return self._caps

    @property
    def default_capabilities(self):
        return {}