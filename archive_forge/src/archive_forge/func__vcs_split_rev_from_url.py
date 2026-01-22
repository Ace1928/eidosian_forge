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
@staticmethod
def _vcs_split_rev_from_url(url, pop_prefix=False):
    scheme, netloc, path, query, frag = urllib.parse.urlsplit(url)
    scheme = scheme.split('+', 1)[-1]
    path = path.split('#', 1)[0]
    rev = None
    if '@' in path:
        path, rev = path.rsplit('@', 1)
    url = urllib.parse.urlunsplit((scheme, netloc, path, query, ''))
    return (url, rev)