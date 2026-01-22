import sys
import io
import random
import mimetypes
import time
import os
import shutil
import smtplib
import shlex
import re
import subprocess
from urllib.parse import urlencode
from urllib import parse as urlparse
from http.cookies import BaseCookie
from paste import wsgilib
from paste import lint
from paste.response import HeaderDict
class FoundDir(object):
    """
    Represents a directory created by a command.
    """
    file = False
    dir = True

    def __init__(self, base_path, path):
        self.base_path = base_path
        self.path = path
        self.full = os.path.join(base_path, path)
        self.size = 'N/A'
        self.mtime = 'N/A'

    def __repr__(self):
        return '<%s %s:%s>' % (self.__class__.__name__, self.base_path, self.path)