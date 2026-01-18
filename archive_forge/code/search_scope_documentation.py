import itertools
import logging
import os
import posixpath
import urllib.parse
from typing import List
from pip._vendor.packaging.utils import canonicalize_name
from pip._internal.models.index import PyPI
from pip._internal.utils.compat import has_tls
from pip._internal.utils.misc import normalize_path, redact_auth_from_url
Returns the locations found via self.index_urls

        Checks the url_name on the main (first in the list) index and
        use this url_name to produce all locations
        