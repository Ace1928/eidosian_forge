import functools
import typing
from base64 import urlsafe_b64decode
from base64 import urlsafe_b64encode
from enum import Enum
@property
def user_handle(self) -> typing.Optional[str]:
    if self._user_handle:
        return urlsafe_b64encode(self._user_handle).decode()
    return None