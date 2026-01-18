import codecs
import copy
from io import BytesIO
from itertools import chain
from urllib.parse import parse_qsl, quote, urlencode, urljoin, urlsplit
from django.conf import settings
from django.core import signing
from django.core.exceptions import (
from django.core.files import uploadhandler
from django.http.multipartparser import (
from django.utils.datastructures import (
from django.utils.encoding import escape_uri_path, iri_to_uri
from django.utils.functional import cached_property
from django.utils.http import is_same_domain, parse_header_parameters
from django.utils.regex_helper import _lazy_re_compile
def validate_host(host, allowed_hosts):
    """
    Validate the given host for this site.

    Check that the host looks valid and matches a host or host pattern in the
    given list of ``allowed_hosts``. Any pattern beginning with a period
    matches a domain and all its subdomains (e.g. ``.example.com`` matches
    ``example.com`` and any subdomain), ``*`` matches anything, and anything
    else must match exactly.

    Note: This function assumes that the given host is lowercased and has
    already had the port, if any, stripped off.

    Return ``True`` for a valid host, ``False`` otherwise.
    """
    return any((pattern == '*' or is_same_domain(host, pattern) for pattern in allowed_hosts))