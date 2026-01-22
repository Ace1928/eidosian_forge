from xmlrpc.client import Fault, dumps, loads, gzip_encode, gzip_decode
from http.server import BaseHTTPRequestHandler
from functools import partial
from inspect import signature
import html
import http.server
import socketserver
import sys
import os
import re
import pydoc
import traceback
def _get_css(self, url):
    path_here = os.path.dirname(os.path.realpath(__file__))
    css_path = os.path.join(path_here, '..', 'pydoc_data', '_pydoc.css')
    with open(css_path, mode='rb') as fp:
        return fp.read()