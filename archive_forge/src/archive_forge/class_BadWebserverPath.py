import errno
import http.client as http_client
import http.server as http_server
import os
import posixpath
import random
import re
import socket
import sys
from urllib.parse import urlparse
from .. import osutils, urlutils
from . import test_server
class BadWebserverPath(ValueError):

    def __str__(self):
        return 'path %s is not in %s' % self.args