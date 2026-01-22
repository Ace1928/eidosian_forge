import abc
import json
import os
from typing import NamedTuple
from google.auth import _helpers
from google.auth import exceptions
from google.auth import external_account
class _TokenContent(NamedTuple):
    """Models the token content response from file and url internal suppliers.
        Attributes:
            content (str): The string content of the file or URL response.
            location (str): The location the content was retrieved from. This will either be a file location or a URL.
    """
    content: str
    location: str