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
def _parse_action(self):
    self.action = None
    text = self.text
    for match in self._tag_re.finditer(text):
        end = match.group(1) == '/'
        tag = match.group(2).lower()
        if tag != 'form':
            continue
        if end:
            break
        attrs = _parse_attrs(match.group(3))
        self.action = attrs.get('action', '')
        self.method = attrs.get('method', 'GET')
        self.id = attrs.get('id')
    else:
        assert 0, 'No </form> tag found'
    assert self.action is not None, 'No <form> tag found'