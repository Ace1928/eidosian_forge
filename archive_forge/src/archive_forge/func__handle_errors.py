import json
from six.moves import urllib
from google_reauth import errors
def _handle_errors(msg):
    """Raise an exception if msg has errors.

    Args:
        msg: parsed json from http response.

    Returns: input response.
    Raises: ReauthAPIError
    """
    if 'error' in msg:
        raise errors.ReauthAPIError(msg['error']['message'])
    return msg