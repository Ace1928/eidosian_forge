from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import json
import sys
from google_reauth import challenges
from google_reauth import errors
from google_reauth import _helpers
from google_reauth import _reauth_client
from six.moves import http_client
from six.moves import range
def _rapt_refresh_required(content):
    """Checks if the rapt refresh is required.

    Args:
        content: refresh response content

    Returns:
        True if rapt refresh is required.
    """
    try:
        content = json.loads(content)
    except (TypeError, ValueError):
        return False
    return content.get('error') == _REAUTH_NEEDED_ERROR and (content.get('error_subtype') == _REAUTH_NEEDED_ERROR_INVALID_RAPT or content.get('error_subtype') == _REAUTH_NEEDED_ERROR_RAPT_REQUIRED)