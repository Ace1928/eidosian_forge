from collections.abc import Sequence
from typing import Optional, Union, cast
from twisted.python.compat import nativeString
from twisted.web._responses import RESPONSES
def _codeToMessage(code: Union[int, bytes]) -> Optional[bytes]:
    """
    Returns the response message corresponding to an HTTP code, or None
    if the code is unknown or unrecognized.

    @param code: HTTP status code, for example C{http.NOT_FOUND}.

    @return: A string message or none
    """
    try:
        return RESPONSES.get(int(code))
    except (ValueError, AttributeError):
        return None