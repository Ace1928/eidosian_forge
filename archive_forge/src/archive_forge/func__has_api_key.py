import time
import warnings
import io
from urllib.error import URLError, HTTPError
from urllib.parse import urlencode
from urllib.request import urlopen, Request
from Bio._utils import function_with_previous
def _has_api_key(request):
    """Check if a Request has the api_key parameter set, to set the rate limit.

    Works with GET or POST requests.
    """
    if request.method == 'POST':
        return b'api_key=' in request.data
    return 'api_key=' in request.full_url