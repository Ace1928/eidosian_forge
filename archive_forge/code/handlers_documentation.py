from __future__ import print_function
import collections
import contextlib
import gzip
import json
import logging
import sys
import time
import zlib
from datetime import datetime, timedelta
from io import BytesIO
from tornado import httputil
from tornado.web import RequestHandler
from urllib3.packages.six import binary_type, ensure_str
from urllib3.packages.six.moves.http_client import responses
from urllib3.packages.six.moves.urllib.parse import urlsplit
Perform a redirect to ``target``