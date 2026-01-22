import email.message
import logging
import pathlib
import traceback
import urllib.parse
import warnings
from typing import Any, Callable, Dict, Iterator, Literal, Optional, Tuple, Type, Union
import requests
from gitlab import types
class EncodedId(str):
    """A custom `str` class that will return the URL-encoded value of the string.

      * Using it recursively will only url-encode the value once.
      * Can accept either `str` or `int` as input value.
      * Can be used in an f-string and output the URL-encoded string.

    Reference to documentation on why this is necessary.

    See::

        https://docs.gitlab.com/ee/api/index.html#namespaced-path-encoding
        https://docs.gitlab.com/ee/api/index.html#path-parameters
    """

    def __new__(cls, value: Union[str, int, 'EncodedId']) -> 'EncodedId':
        if isinstance(value, EncodedId):
            return value
        if not isinstance(value, (int, str)):
            raise TypeError(f'Unsupported type received: {type(value)}')
        if isinstance(value, str):
            value = urllib.parse.quote(value, safe='')
        return super().__new__(cls, value)