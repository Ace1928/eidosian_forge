import locale
from typing import TYPE_CHECKING, Any, Dict, Optional, Union
from .namespaces import XQT_ERRORS_NAMESPACE
from .datatypes import QName
class ElementPathError(Exception):
    """
    Base exception class for elementpath package.

    :param message: the message related to the error.
    :param code: an optional error code.
    :param token: an optional token instance related with the error.
    """

    def __init__(self, message: str, code: Optional[str]=None, token: Optional['Token[Any]']=None) -> None:
        super(ElementPathError, self).__init__(message)
        self.message = message
        self.code = code
        self.token = token

    def __str__(self) -> str:
        if self.token is None or not isinstance(self.token.value, (str, bytes)):
            if not self.code:
                return self.message
            return '[{}] {}'.format(self.code, self.message)
        elif not self.code:
            return '{1} at line {2}, column {3}: {0}'.format(self.message, self.token, *self.token.position)
        return '{2} at line {3}, column {4}: [{1}] {0}'.format(self.message, self.code, self.token, *self.token.position)