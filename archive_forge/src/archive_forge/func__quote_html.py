from __future__ import (absolute_import, division,
from future import utils
from future.builtins import *
from future.backports import html
from future.backports.http import client as http_client
from future.backports.urllib import parse as urllib_parse
from future.backports import socketserver
import io
import mimetypes
import os
import posixpath
import select
import shutil
import socket # For gethostbyaddr()
import sys
import time
import copy
import argparse
def _quote_html(html):
    return html.replace('&', '&amp;').replace('<', '&lt;').replace('>', '&gt;')