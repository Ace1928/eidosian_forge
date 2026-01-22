import sys
import os
import re
import io
import shutil
import socket
import base64
import hashlib
import itertools
import configparser
import html
import http.client
import urllib.parse
import urllib.request
import urllib.error
from functools import wraps
import setuptools
from pkg_resources import (
from distutils import log
from distutils.errors import DistutilsError
from fnmatch import translate
from setuptools.wheel import Wheel
from setuptools.extern.more_itertools import unique_everseen
def _attempt_download(self, url, filename):
    headers = self._download_to(url, filename)
    if 'html' in headers.get('content-type', '').lower():
        return self._invalid_download_html(url, headers, filename)
    else:
        return filename