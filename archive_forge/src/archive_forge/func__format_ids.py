import time
import warnings
import io
from urllib.error import URLError, HTTPError
from urllib.parse import urlencode
from urllib.request import urlopen, Request
from Bio._utils import function_with_previous
def _format_ids(ids):
    """Convert one or more UIDs to a single comma-delimited string.

    Input may be a single ID as an integer or string, an iterable of strings/ints,
    or a string of IDs already separated by commas.
    """
    if isinstance(ids, int):
        return str(ids)
    if isinstance(ids, str):
        return ','.join((id.strip() for id in ids.split(',')))
    return ','.join(map(str, ids))