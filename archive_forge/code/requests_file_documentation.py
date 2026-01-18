from requests.adapters import BaseAdapter
from requests.compat import urlparse, unquote
from requests import Response, codes
import errno
import os
import stat
import locale
import io
Wraps a file, described in request, in a Response object.

        :param request: The PreparedRequest` being "sent".
        :returns: a Response object containing the file
        