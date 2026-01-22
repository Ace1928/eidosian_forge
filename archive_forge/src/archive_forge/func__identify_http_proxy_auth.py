import logging
import os
import platform
import socket
import string
from base64 import b64encode
from urllib import parse
import certifi
import urllib3
from selenium import __version__
from . import utils
from .command import Command
from .errorhandler import ErrorCode
def _identify_http_proxy_auth(self):
    url = self._proxy_url
    url = url[url.find(':') + 3:]
    return '@' in url and len(url[:url.find('@')]) > 0