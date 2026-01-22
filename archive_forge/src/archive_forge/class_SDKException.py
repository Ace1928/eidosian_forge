import json
import re
import typing as ty
from requests import exceptions as _rex
class SDKException(Exception):
    """The base exception class for all exceptions this library raises."""

    def __init__(self, message=None, extra_data=None):
        self.message = self.__class__.__name__ if message is None else message
        self.extra_data = extra_data
        super(SDKException, self).__init__(self.message)