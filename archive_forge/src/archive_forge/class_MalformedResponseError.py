from enum import Enum
from typing import Union, Callable, Optional, cast
class MalformedResponseError(LibcloudError):
    """Exception for the cases when a provider returns a malformed
    response, e.g. you request JSON and provider returns
    '<h3>something</h3>' due to some error on their side."""

    def __init__(self, value, body=None, driver=None):
        self.value = value
        self.driver = driver
        self.body = body

    def __str__(self):
        return self.__repr__()

    def __repr__(self):
        return '<MalformedResponseException in ' + repr(self.driver) + ' ' + repr(self.value) + '>: ' + repr(self.body)