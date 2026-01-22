import copy
import datetime
import json
import cachetools
import six
from six.moves import urllib
from google.auth import _helpers
from google.auth import _service_account_info
from google.auth import crypt
from google.auth import exceptions
import google.auth.credentials
def _decode_jwt_segment(encoded_section):
    """Decodes a single JWT segment."""
    section_bytes = _helpers.padded_urlsafe_b64decode(encoded_section)
    try:
        return json.loads(section_bytes.decode('utf-8'))
    except ValueError as caught_exc:
        new_exc = exceptions.MalformedError("Can't parse segment: {0}".format(section_bytes))
        six.raise_from(new_exc, caught_exc)