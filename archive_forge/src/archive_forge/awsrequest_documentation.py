import functools
import logging
from collections.abc import Mapping
import urllib3.util
from urllib3.connection import HTTPConnection, VerifiedHTTPSConnection
from urllib3.connectionpool import HTTPConnectionPool, HTTPSConnectionPool
import botocore.utils
from botocore.compat import (
from botocore.exceptions import UnseekableStreamError
Content of the response as a proper text type.

        Uses the encoding type provided in the reponse headers to decode the
        response content into a proper text type. If the encoding is not
        present in the headers, UTF-8 is used as a default.
        