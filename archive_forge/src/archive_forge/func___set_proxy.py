import os
import time
import getpass
import json
import requests
from urllib.parse import urlparse
from .select import Select
from .help import Inspector, GraphData, PaintGraph, _DRAW_GRAPHS
from .manage import GlobalManager
from .uriutil import join_uri, file_path, uri_last
from .jsonutil import csv_to_json
from .errors import is_xnat_error
from .errors import catch_error
from .array import ArrayData
from .xpath_store import XpathStore
from . import xpass
def __set_proxy(self, proxy=None):
    if proxy is None:
        proxy = os.environ.get('http_proxy')
    if proxy is None:
        self._proxy_url = None
    else:
        self._proxy_url = urlparse(proxy)