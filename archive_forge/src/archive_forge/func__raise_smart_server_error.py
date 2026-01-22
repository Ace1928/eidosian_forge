from collections import deque
from io import BytesIO
from ... import debug, errors
from ...trace import mutter
def _raise_smart_server_error(error_tuple):
    """Raise exception based on tuple received from smart server

    Specific error translation is handled by breezy.bzr.remote._translate_error
    """
    if error_tuple[0] == b'UnknownMethod':
        raise errors.UnknownSmartMethod(error_tuple[1])
    raise errors.ErrorFromSmartServer(error_tuple)